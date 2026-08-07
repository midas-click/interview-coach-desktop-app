# Notepadder — Real-time Interview Transcriber

Windows desktop application for capturing and transcribing live job interviews. Captures microphone and system audio, transcribes with Whisper, persists to SQLite, and uploads to S3 for the Interview Intelligence analysis pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     UI (PySide6)                            │
│  MainWindow → Dashboard | Controls | TranscriptView         │
└──────────────────────┬──────────────────────────────────────┘
                       │ Qt signals + asyncio (qasync)
┌──────────────────────▼──────────────────────────────────────┐
│               MeetingController (orchestrator)              │
└──┬──────────────┬──────────────────┬───────────────────────┘
   │              │                  │
┌──▼──────────┐ ┌─▼───────────────┐ ┌▼──────────────┐
│ AudioManager│ │Transcription    │ │ Repository    │
│             │ │Engine           │ │ (SQLite)      │
│ mic source  │ │                 │ │                │
│ sys source  │ │ queue→whisper   │ │ insert chunks │
│ audio→queue │ │ →segment_builder│ │ every 3s       │
└─────────────┘ └─────────────────┘ └───────────────┘
                       │
                  ┌────▼──────┐
                  │ S3Uploader│
                  │ retry 3x  │
                  └───────────┘
```

### Module Map

| Module | File | Responsibility |
|--------|------|---------------|
| config | `src/config/settings.py` | Pydantic model, YAML load/save, env overrides |
| logger | `src/logger/logger.py` | Structured logging (console + JSON file) |
| audio | `src/audio/capture.py`, `manager.py` | sounddevice capture (WASAPI native), float32 numpy output |
| transcription | `src/transcription/engine.py`, `segment_builder.py` | faster-whisper in daemon thread, sentence merging |
| storage | `src/storage/models.py`, `repository.py` | SQLite with WAL, async CRUD, JSON/TXT export |
| meeting | `src/meeting/controller.py` | Lifecycle orchestrator, flush loop, crash recovery |
| upload | `src/upload/s3_uploader.py` | S3 upload with 3-retry exponential backoff |
| ui | `src/ui/*.py` | PySide6: dashboard, controls, transcript, settings |

---

## Setup

### Prerequisites

- **Python 3.12+**
- **Working microphone** (USB/headset mic recommended).
- **System audio loopback** — to capture speaker output alongside the mic,
  Windows must expose a WASAPI loopback device for your output device.
  Most audio interfaces expose one automatically. Check with the app's
  device dropdown after first launch.

  Audio capture uses `sounddevice` (WASAPI native) — no external codecs required.

### Install

```powershell
# Clone
git clone <repo-url>
cd desktop-app

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "from src.main import main; print('Ready')"
```

### Configure

1. Copy the example files:

   ```powershell
   copy config.yaml.example config.yaml
   copy .env.example .env
   ```

2. Edit `config.yaml` — choose your whisper model, output directory, etc.

3. (Optional) Set AWS credentials in `.env` if you want S3 upload:

   ```
   AWS_ACCESS_KEY_ID=AKIA...
   AWS_SECRET_ACCESS_KEY=...
   AWS_REGION=us-east-2
   AWS_BUCKET=my-company-transcripts
   ```

   Environment variables override `config.yaml` for secrets.

---

## Usage

```powershell
python -m src.main
```

### Workflow

1. **Launch** — the app checks for an unfinished meeting (crash recovery).
2. **Enter** company name and interview stage (optional).
3. **Click ▶ Start** — audio capture begins. Live transcription appears below.
4. Use **⏸ Pause** / **▶ Resume** as needed during the interview.
5. **Click ⏹ Finish** when done:
   - Transcription finalises
   - `transcript.json` + `transcript.txt` saved to `output_dir/<meeting_id>/`
   - JSON uploaded to S3 (if auto-upload is on and credentials are set)
   - UI returns to idle state

### Settings (⚙)

| Setting | Description |
|---------|-------------|
| Whisper model | `tiny` through `large-v3`. Smaller = faster, lower accuracy. |
| Microphone | Input device index. Leave as `(default)` for system default. |
| System audio | Loopback device for capturing speaker output. |
| Output directory | Where JSON/TXT exports are saved. |
| Auto-upload to S3 | Upload JSON automatically on finish. |
| AWS region / bucket / keys | S3 credentials. |
| Log level | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |

---

## Output Format

### JSON (`transcript.json`)

```json
{
  "meetingId": "a1b2c3d4e5f6",
  "companyName": "Google",
  "interviewStage": "Coding Interview",
  "createdAt": "2026-01-15T14:30:00+00:00",
  "language": "en",
  "transcript": [
    {
      "speaker": "Unknown",
      "start": 12.5,
      "end": 15.2,
      "confidence": 0.93,
      "text": "Tell me about yourself."
    }
  ]
}
```

### TXT (`transcript.txt`)

```
Meeting: a1b2c3d4e5f6
Company: Google
Stage: Coding Interview
Date: 2026-01-15 14:30 UTC

[00:00:12] Unknown: Tell me about yourself.
[00:00:16] Unknown: I have 10 years of experience.
```

### S3 Path

```
s3://<bucket>/interviews/<meeting_id>/transcript.json
```

---

## Database

SQLite file at `data/transcriber.db` (WAL mode, auto-created).

**Tables:**

- `meetings` — meeting_id, company_name, interview_stage, created_at, started_at, ended_at, status
- `transcript_chunks` — id, meeting_id, speaker, start_time, end_time, confidence, text

Crash recovery: on startup, the app finds any meeting with `status = 'active' | 'paused'` and offers to resume.

---

## Development

### Run Tests

```powershell
pytest tests/ -v
```

32 tests across 4 modules (audio, transcription, meeting, upload).

### Project Conventions

- **Type hints** everywhere
- **Pydantic** for data models
- **Async** where I/O is involved (SQLite, S3)
- **Thread-safe queues** bridge audio (callback thread) → transcription (worker thread) → UI (Qt main thread)
- **qasync** merges asyncio event loop with Qt event loop — no thread-pool hacks

### Adding Features

Architecture supports future extensions without major refactoring:

- **Speaker diarization** — add a `diarization/` module that tags `TranscriptChunk.speaker`
- **Live translation** — add `translation/` module as a post-processing step
- **AI coaching** — subscribe to `on_chunks_persisted` callback, add a coach module
- **Streaming upload** — convert periodic flush loop to also upload chunks incrementally

---

## Standalone Build

```powershell
.\build.ps1
```

Outputs `dist/InterviewTranscriber/`. Copy that folder to any Windows 11
machine and run `InterviewTranscriber.exe` — no Python or other dependencies
needed. The bundled whisper model is auto-detected at startup.