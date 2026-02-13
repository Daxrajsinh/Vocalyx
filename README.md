# Vocalyx

Real-time voice assistant: Speech-to-Text (STT) → GPT → Text-to-Speech (TTS). Install with `pip install vocalyx`. Run `vocalyx-stt`, `vocalyx-tts`, and `vocalyx` in three terminals.

---

## Install

```bash
pip install vocalyx
```

**Requirements:** Python 3.10+, system dependencies `portaudio` and `ffmpeg`, microphone and audio output.

---

## Architecture and flow

All components live in the `vocalyx` package and talk over WebSockets:

```
[Microphone] → vocalyx (client)
                    ↓
              STT server (RealtimeSTT) ← control + data WebSockets
                    ↓
              vocalyx (client) receives transcribed text
                    ↓
              TTS server (GPT + Soprano TTS) ← single WebSocket
                    ↓
              vocalyx (client) plays audio → [Speakers]
```

1. **vocalyx-stt** – STT server. Listens on two ports: control (commands) and data (audio + transcriptions).
2. **vocalyx-tts** – TTS server. Listens on one port. Receives text, streams GPT response, synthesizes with Soprano, streams audio back.
3. **vocalyx** – Client. Captures mic, sends audio to STT; receives text, sends to TTS; receives and plays TTS audio.

---

## Ports (defaults)

| Component   | Role              | Default URL / Port        |
|------------|-------------------|---------------------------|
| STT server | Control WebSocket | `ws://localhost:8011`     |
| STT server | Data WebSocket    | `ws://localhost:8012`     |
| TTS server | WebSocket         | `ws://localhost:8013`     |

The client uses these URLs by default. You can override them with `--control`, `--data`, and `--tts-url` when running `vocalyx`, and with `-c`/`-d` when running `vocalyx-stt`. The TTS server binds to `localhost` and the port above (no CLI port flag in the current release).

---

## Models used

| Role | Model / service      | Notes |
|------|----------------------|--------|
| **STT** | RealtimeSTT (Faster Whisper, CTranslate2) | Main model default: `tiny.en`. Real-time model default: `tiny`. Configurable via `-m`, `-r`. |
| **LLM** | OpenAI GPT (streaming) | Default: `gpt-3.5-turbo`. Used for assistant replies. Requires `OPENAI_API_KEY`. |
| **TTS** | Soprano TTS | On-device, 80M params. Default: `backend=auto`, `device=auto`, `cache_size_mb=100`, `decoder_batch_size=1`. Output: 32 kHz mono float32. |

---

## Configuration

### Environment

Create a `.env` in the directory from which you run the processes (or set the variable in the shell):

- **`OPENAI_API_KEY`** – Required for the TTS server (GPT). Get it from [OpenAI API keys](https://platform.openai.com/api-keys).

### TTS server (`vocalyx-tts`)

- **Host / port:** `localhost:8013` (hardcoded in code).
- **GPT:** System prompt and model are sent by the client per request (default prompt “motivational assistant”.", default model: `gpt-3.5-turbo`). Temperature is `0.7`.
- **Soprano:** Loaded once at startup; `backend="auto"`, `device="auto"`, `cache_size_mb=100`, `decoder_batch_size=1`.

### STT server (`vocalyx-stt`)

Key options (run `vocalyx-stt --help` for full list):

| Option | Default | Description |
|--------|---------|-------------|
| `-c`, `--control` | 8011 | Control WebSocket port. |
| `-d`, `--data` | 8012 | Data WebSocket port. |
| `-m`, `--model` | `tiny.en` | Main Whisper model size or path (e.g. `base`, `small`, `large-v2`). |
| `-r`, `--rt-model` | `tiny` | Real-time transcription model size. |
| `-l`, `--lang` | `en` | Language code. |
| `-i`, `--input-device` | 1 | Audio input device index. |
| `--device` | `cuda` | `cuda` or `cpu`. |
| `--compute_type` | `default` | CTranslate2 compute type. |
| `--silero_sensitivity` | 0.05 | VAD sensitivity (0–1). |
| `--webrtc_sensitivity` | 3 | WebRTC VAD (0–3). |
| `--end_of_sentence_detection_pause` | 0.45 | Silence (seconds) to treat as end of sentence. |
| `-D`, `--debug` | off | Debug logging. |

### Client (`vocalyx`)

| Option | Default | Description |
|--------|---------|-------------|
| `--control` | `ws://localhost:8011` | STT control WebSocket URL. |
| `--data` | `ws://localhost:8012` | STT data WebSocket URL. |
| `--tts-url` | `ws://localhost:8013` | TTS WebSocket URL. |
| `--voice` | `af_heart` | Voice name (Soprano may ignore). |
| `--speed` | 1.0 | Playback speed. |
| `--system-prompt`, `--prompt` | `You are a motivational assistant.` | LLM system prompt (persona/instructions). |
| `--model` | `gpt-3.5-turbo` | OpenAI model (e.g. `gpt-4`, `gpt-4o`). |
| `-L`, `--list` | - | List audio devices and exit. |
| `-c`, `--continous` | true | Keep running and transcribing. |

---

## Usage

1. Set `OPENAI_API_KEY` (e.g. in `.env`).
2. Start STT server, then TTS server, then client:

```bash
# Terminal 1
vocalyx-stt

# Terminal 2
vocalyx-tts

# Terminal 3
vocalyx
```

3. Speak; the client shows “You” and “AI” and plays the assistant’s voice.

List microphone devices:

```bash
vocalyx --list
```

Use a different TTS URL or STT URLs if your servers run on other hosts/ports:

```bash
vocalyx --tts-url ws://otherhost:8013 --control ws://otherhost:8011 --data ws://otherhost:8012
```

---

## Audio format (TTS)

Soprano TTS outputs **32 kHz, mono, float32**. The client plays this directly (e.g. `paFloat32`, 32000 Hz, 1 channel). Do not convert to μ-law or int16 for playback or you may change pitch/tempo.

---

## License

MIT
