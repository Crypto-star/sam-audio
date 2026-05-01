import modal
import os
import uuid
import time
import threading
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("sam-audio-service")

hf_secret = modal.Secret.from_name("huggingface-secret")
model_volume = modal.Volume.from_name("sam-audio-model-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image definition
# ---------------------------------------------------------------------------
sam_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1", "build-essential", "pkg-config")
    .pip_install(
        "torch", "torchaudio", "torchvision", "torchcodec",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.54.0",
        "torchdiffeq",
        "einops",
        "numpy",
        "pydub",
        "audiobox_aesthetics",
        "soundfile",
        "huggingface_hub",
    )
    .pip_install("dacvae @ git+https://github.com/facebookresearch/dacvae.git")
    .pip_install("imagebind @ git+https://github.com/facebookresearch/ImageBind.git")
    .pip_install("laion-clap @ git+https://github.com/lematt1991/CLAP.git")
    .pip_install(
        "perception-models @ git+https://github.com/facebookresearch/perception_models@unpin-deps"
    )
    .pip_install("fastapi", "python-multipart")
    .add_local_dir("./repo", "/sam_audio_repo", copy=True)
    .run_commands("cd /sam_audio_repo && pip install --no-deps .")
    .env({"HF_HOME": "/model_cache", "TORCH_HOME": "/model_cache"})
    .add_local_dir("./frontend", "/app")
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48_000
HOP_LENGTH = 1920  # prod([2, 8, 10, 12])
CHUNK_SECONDS = 60
OVERLAP_SECONDS = 5
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE
OVERLAP_SAMPLES = OVERLAP_SECONDS * SAMPLE_RATE
STEP_SAMPLES = CHUNK_SAMPLES - OVERLAP_SAMPLES

SPEAKER_PRESETS = {
    "male_voice": "man speaking",
    "female_voice": "woman speaking",
    "child_voice": "child speaking",
    "narrator": "narrator speaking",
    "crowd": "crowd talking",
    "singing_male": "man singing",
    "singing_female": "woman singing",
    "music": "music",
    "speech": "speech",
}

MODEL_SIZES = {"small", "base", "large"}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
@app.cls(
    image=sam_image,
    gpu="A100-80GB",
    volumes={"/model_cache": model_volume},
    secrets=[hf_secret],
    scaledown_window=300,
    timeout=3600,
)
@modal.concurrent(max_inputs=100)
class SAMAudioService:

    @modal.enter()
    def setup(self):
        self._models: dict = {}
        self._jobs: dict = {}
        self._gpu_lock = threading.Semaphore(1)
        os.makedirs("/tmp/sam_jobs", exist_ok=True)

    # -- model management ---------------------------------------------------

    def _get_model(self, size: str):
        if size not in self._models:
            import torch
            from sam_audio import SAMAudio, SAMAudioProcessor

            repo_id = f"facebook/sam-audio-{size}"
            print(f"[sam-audio] loading {repo_id} …")
            model = SAMAudio.from_pretrained(repo_id).eval().cuda()
            processor = SAMAudioProcessor.from_pretrained(repo_id)
            self._models[size] = (model, processor)
            model_volume.commit()
            print(f"[sam-audio] {repo_id} ready")
        return self._models[size]

    # -- chunking -----------------------------------------------------------

    @staticmethod
    def _compute_chunks(total_samples: int) -> list[tuple[int, int]]:
        if total_samples <= CHUNK_SAMPLES:
            return [(0, total_samples)]
        chunks = []
        pos = 0
        while pos < total_samples:
            start = pos
            end = min(pos + CHUNK_SAMPLES, total_samples)
            chunks.append((start, end))
            if end >= total_samples:
                break
            pos += STEP_SAMPLES
        return chunks

    @staticmethod
    def _stitch(chunks: list, boundaries: list[tuple[int, int]], total: int):
        import torch

        out = torch.zeros(total)
        wgt = torch.zeros(total)

        for i, (wav, (s, e)) in enumerate(zip(chunks, boundaries)):
            length = e - s
            fade = torch.ones(length)
            if i > 0:
                fl = min(OVERLAP_SAMPLES, length)
                fade[:fl] = torch.linspace(0, 1, fl)
            if i < len(chunks) - 1:
                fr = min(OVERLAP_SAMPLES, length)
                fade[-fr:] = torch.linspace(1, 0, fr)
            out[s:e] += wav[:length] * fade
            wgt[s:e] += fade

        return out / wgt.clamp(min=1e-8)

    # -- single-chunk separation --------------------------------------------

    def _run_chunk(self, model, processor, audio, description, predict_spans, candidates):
        import torch

        batch = processor(audios=[audio], descriptions=[description]).to("cuda")
        result = model.separate(
            batch,
            predict_spans=predict_spans,
            reranking_candidates=candidates,
        )
        t = result.target[0].cpu()
        r = result.residual[0].cpu()
        del batch, result
        torch.cuda.empty_cache()
        return t, r

    # -- full job processing (runs in background thread) --------------------

    def _process_job(self, job_id, audio_bytes, filename, description, model_size, quality):
        import torch
        import torchaudio

        job = self._jobs[job_id]
        try:
            job.update(status="loading_model", message=f"Loading {model_size} model …")
            model, processor = self._get_model(model_size)

            job.update(status="loading_audio", message="Decoding audio …")
            suffix = Path(filename).suffix or ".wav"
            tmp_in = f"/tmp/sam_jobs/{job_id}_in{suffix}"
            with open(tmp_in, "wb") as f:
                f.write(audio_bytes)

            wav, sr = torchaudio.load(tmp_in, backend="ffmpeg")
            os.remove(tmp_in)

            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            wav = wav.mean(0, keepdim=True)  # (1, samples)
            total = wav.shape[-1]
            dur = total / SAMPLE_RATE

            predict_spans = quality == "high"
            candidates = 4 if quality == "high" else 1

            boundaries = self._compute_chunks(total)
            n = len(boundaries)

            job.update(
                status="processing",
                message=f"Separating {dur:.0f}s audio ({n} chunk{'s' if n > 1 else ''}) …",
                total_chunks=n,
            )

            target_parts, residual_parts = [], []

            for i, (s, e) in enumerate(boundaries):
                job.update(
                    current_chunk=i + 1,
                    progress=int(i / n * 90),
                    message=f"Chunk {i + 1}/{n} …",
                )

                chunk = wav[:, s:e]
                clen = chunk.shape[-1]
                if clen % HOP_LENGTH:
                    chunk = torch.nn.functional.pad(chunk, (0, HOP_LENGTH - clen % HOP_LENGTH))

                with self._gpu_lock:
                    t, r = self._run_chunk(model, processor, chunk, description, predict_spans, candidates)
                target_parts.append(t[:clen])
                residual_parts.append(r[:clen])

            job.update(progress=92, message="Stitching …")

            if n == 1:
                final_t, final_r = target_parts[0], residual_parts[0]
            else:
                final_t = self._stitch(target_parts, boundaries, total)
                final_r = self._stitch(residual_parts, boundaries, total)

            job.update(progress=96, message="Saving …")

            out_dir = f"/tmp/sam_jobs/{job_id}"
            os.makedirs(out_dir, exist_ok=True)
            torchaudio.save(f"{out_dir}/target.wav", final_t.unsqueeze(0), SAMPLE_RATE)
            torchaudio.save(f"{out_dir}/residual.wav", final_r.unsqueeze(0), SAMPLE_RATE)

            job.update(status="done", progress=100, message="Done!", duration_s=round(dur, 1))

        except Exception as exc:
            import traceback
            traceback.print_exc()
            job.update(status="error", progress=0, message=str(exc))

    # -- ASGI app -----------------------------------------------------------

    @modal.asgi_app()
    def web(self):
        import json
        import asyncio
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException
        from fastapi.responses import FileResponse, Response, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware

        api = FastAPI(title="SAM-Audio Service")
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        svc = self

        @api.get("/")
        async def index():
            return FileResponse("/app/index.html", media_type="text/html")

        @api.get("/api/presets")
        async def presets():
            return SPEAKER_PRESETS

        @api.post("/api/separate")
        async def separate(
            audio: UploadFile = File(...),
            description: str = Form(""),
            preset: str = Form(""),
            model_size: str = Form("large"),
            quality_mode: str = Form("fast"),
        ):
            if preset and preset in SPEAKER_PRESETS:
                desc = SPEAKER_PRESETS[preset]
            elif description.strip():
                desc = description.strip().lower()
            else:
                raise HTTPException(400, "Provide a text prompt or select a speaker preset")

            if model_size not in MODEL_SIZES:
                raise HTTPException(400, f"model_size must be one of {MODEL_SIZES}")
            if quality_mode not in ("fast", "high"):
                raise HTTPException(400, "quality_mode must be 'fast' or 'high'")

            raw = await audio.read()
            if len(raw) > 2 * 1024**3:
                raise HTTPException(400, "File exceeds 2 GB limit")

            jid = str(uuid.uuid4())
            svc._jobs[jid] = dict(
                status="queued",
                progress=0,
                message="Queued …",
                description=desc,
                model_size=model_size,
                quality_mode=quality_mode,
                created_at=time.time(),
            )

            threading.Thread(
                target=svc._process_job,
                args=(jid, raw, audio.filename or "audio.wav", desc, model_size, quality_mode),
                daemon=True,
            ).start()

            return {"job_id": jid}

        @api.get("/api/jobs/{job_id}")
        async def job_status(job_id: str):
            if job_id not in svc._jobs:
                raise HTTPException(404, "Job not found")
            return svc._jobs[job_id]

        @api.get("/api/jobs/{job_id}/events")
        async def job_events(job_id: str):
            if job_id not in svc._jobs:
                raise HTTPException(404, "Job not found")

            async def stream():
                while True:
                    j = svc._jobs.get(job_id, {})
                    payload = json.dumps(
                        {k: j.get(k) for k in ("status", "progress", "message")}
                    )
                    yield f"data: {payload}\n\n"
                    if j.get("status") in ("done", "error"):
                        break
                    await asyncio.sleep(0.5)

            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @api.get("/api/jobs/{job_id}/download/{track}")
        async def download(job_id: str, track: str):
            if job_id not in svc._jobs:
                raise HTTPException(404, "Job not found")
            if track not in ("target", "residual"):
                raise HTTPException(400, "track must be 'target' or 'residual'")
            if svc._jobs[job_id].get("status") != "done":
                raise HTTPException(400, "Job not complete yet")

            path = f"/tmp/sam_jobs/{job_id}/{track}.wav"
            if not os.path.exists(path):
                raise HTTPException(404, "Result file missing")

            return FileResponse(path, media_type="audio/wav", filename=f"{track}.wav")

        @api.delete("/api/jobs/{job_id}")
        async def delete_job(job_id: str):
            job_dir = f"/tmp/sam_jobs/{job_id}"
            if os.path.isdir(job_dir):
                shutil.rmtree(job_dir, ignore_errors=True)
            svc._jobs.pop(job_id, None)
            return {"ok": True}

        return api
