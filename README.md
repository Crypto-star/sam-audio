# SAM-Audio Separation Service

A web service for audio source separation powered by [Meta's SAM-Audio](https://github.com/facebookresearch/sam-audio), deployed on [Modal](https://modal.com) with GPU acceleration.

Upload an audio file, describe what you want to extract (e.g. "woman speaking", "guitar", "crowd talking"), and get back the separated target audio and residual.

## Features

- **Text-prompted separation** -- describe the sound you want to isolate
- **Speaker presets** -- quick selection for common targets (male/female voice, music, singing, etc.)
- **Long audio support** -- handles 30+ minute files via chunked processing with crossfade stitching
- **Model switching** -- choose between `small`, `base`, and `large` SAM-Audio models
- **Quality modes** -- fast (single pass) or high (span prediction + 4x reranking candidates)
- **Real-time progress** -- SSE-based progress updates during processing
- **Web UI** -- dark-themed frontend with drag-and-drop upload, waveform visualization, and in-browser playback

## Prerequisites

- Python 3.11+
- A [Modal](https://modal.com) account (free tier available)
- A [Hugging Face](https://huggingface.co) account with access to the gated `facebook/sam-audio-*` models

### Hugging Face Model Access

SAM-Audio models are gated. You need to request access before deploying:

1. Go to [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large) on Hugging Face
2. Click **"Agree and access repository"**
3. Repeat for [facebook/sam-audio-base](https://huggingface.co/facebook/sam-audio-base) and [facebook/sam-audio-small](https://huggingface.co/facebook/sam-audio-small) if you want all model sizes
4. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (needs `read` permission)

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Crypto-star/sam-audio.git
cd sam-audio
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows
```

### 3. Install Modal

```bash
pip install modal
```

### 4. Authenticate with Modal

```bash
modal token set --token-id <your-modal-token-id> --token-secret <your-modal-token-secret>
```

You can find your token at [modal.com/settings](https://modal.com/settings). If you're using a Modal profile other than `default`, set the environment variable:

```bash
export MODAL_PROFILE=<your-profile-name>
```

### 5. Create the Hugging Face secret in Modal

SAM-Audio needs your HF token to download the gated models. Create a Modal secret named `huggingface-secret`:

```bash
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>
```

## Deploy

From the project root:

```bash
modal deploy modal_app.py
```

The first deploy takes 5-10 minutes (building the container image with PyTorch, SAM-Audio, and all dependencies). Subsequent deploys reuse cached layers and take ~30 seconds.

Once deployed, Modal prints your service URL:

```
Created web endpoint for SAMAudioService.web =>
    https://<your-workspace>--sam-audio-service-samaudioservice-web.modal.run
```

Open that URL in your browser to use the service.

## Usage

### Web UI

1. Open the deployed URL in your browser
2. Upload an audio file (drag-and-drop or click to browse) -- supports WAV, MP3, FLAC, M4A, OGG, etc.
3. Either:
   - Select a **preset** from the dropdown (e.g. "Male Voice", "Music")
   - Or type a **custom text prompt** (e.g. "piano", "bird chirping", "man with deep voice")
4. Choose a **model size** (large = best quality, small = fastest)
5. Choose **quality mode** (fast or high)
6. Click **Separate** and wait for processing
7. Listen to the results and download the target/residual WAV files

### API

You can also use the API directly:

```bash
# Start a separation job
curl -X POST https://<your-url>/api/separate \
  -F "audio=@recording.mp3" \
  -F "description=woman speaking" \
  -F "model_size=large" \
  -F "quality_mode=fast"
# Returns: {"job_id": "uuid"}

# Poll job status
curl https://<your-url>/api/jobs/<job_id>

# Stream progress via SSE
curl https://<your-url>/api/jobs/<job_id>/events

# Download results (once status is "done")
curl -o target.wav https://<your-url>/api/jobs/<job_id>/download/target
curl -o residual.wav https://<your-url>/api/jobs/<job_id>/download/residual

# Clean up
curl -X DELETE https://<your-url>/api/jobs/<job_id>

# List available presets
curl https://<your-url>/api/presets
```

## Architecture

```
modal_app.py          # Modal backend -- GPU service, chunking, FastAPI endpoints
frontend/index.html   # Single-file web UI (HTML + CSS + JS)
repo/                 # Meta's SAM-Audio source (installed into container)
```

### How it works

1. **Upload**: Audio file is uploaded to the Modal container via FastAPI
2. **Chunking**: Long audio is split into 60-second chunks with 5-second overlap (hop-aligned to 1920 samples)
3. **Separation**: Each chunk is processed by SAM-Audio's diffusion model on A100 GPU
4. **Stitching**: Chunks are reassembled using symmetric crossfade to eliminate seams
5. **Output**: Target (extracted sound) and residual (everything else) are saved as 48kHz WAV

### GPU and scaling

- Default GPU: `A100-80GB` (handles the large model comfortably)
- Container scales to zero after 5 minutes of inactivity (`scaledown_window=300`)
- Job timeout: 1 hour (`timeout=3600`)
- Concurrent requests are handled within a single container via `@modal.concurrent`

To change the GPU type, edit the `gpu` parameter in `modal_app.py`:

```python
@app.cls(
    gpu="A100-80GB",  # or "A10G", "T4", "H100", etc.
    ...
)
```

Smaller GPUs (A10G, T4) work with the `small` and `base` models but may OOM on `large`.

## Configuration

### Model volume

Models are cached in a Modal volume (`sam-audio-model-cache`) so they persist across container restarts. The first run downloads the model from Hugging Face (~2-5 GB per model size); subsequent runs load from cache.

### Speaker presets

Edit the `SPEAKER_PRESETS` dict in `modal_app.py` to customize:

```python
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
```

### Chunking parameters

For very long audio or different GPU memory constraints:

```python
CHUNK_SECONDS = 60      # seconds per chunk (reduce for less VRAM)
OVERLAP_SECONDS = 5     # overlap between chunks (for seamless stitching)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `401 Unauthorized` from Hugging Face | Ensure your HF token has access to the gated `facebook/sam-audio-*` models |
| Container OOM | Switch to a larger GPU or reduce `CHUNK_SECONDS` |
| `BaseModel._from_pretrained() missing arguments` | Already patched in this repo -- newer `huggingface_hub` versions changed the API |
| SSE returns 404 for job | Ensure `@modal.concurrent(max_inputs=100)` is present -- without it, requests hit different containers |
| Slow first request | Cold start downloads model weights; subsequent requests use the cached volume |

## License

The SAM-Audio model and source code in `repo/` are from Meta and subject to their [license](repo/LICENSE). The service wrapper (`modal_app.py`, `frontend/`) is provided as-is.
