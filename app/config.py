"""Central config for the football commentary pipeline. Env-driven, .env auto-loaded."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

_MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models")))


class Settings:
    # LLM
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Paths
    models_dir: Path = _MODELS_DIR
    data_dir: Path = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
    out_root: Path = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output")))

    # Binaries
    ffmpeg: str = os.getenv("FFMPEG_BIN", "ffmpeg")
    ffprobe: str = os.getenv("FFPROBE_BIN", "ffprobe")

    # Speech
    tts_voice: str = os.getenv("TTS_VOICE", "en-US-GuyNeural")
    tts_voice_b: str = os.getenv("TTS_VOICE_B", "en-US-ChristopherNeural")
    tts_rate: str = os.getenv("TTS_RATE", "+10%")  # energetic
    original_audio_volume: float = float(os.getenv("ORIG_AUDIO_VOL", "0.12"))
    voice_boost: float = float(os.getenv("VOICE_BOOST", "2.0"))  # multiplier for commentary voice

    # Vision
    analysis_fps: float = float(os.getenv("ANALYSIS_FPS", "4"))
    analysis_width: int = int(os.getenv("ANALYSIS_WIDTH", "512"))
    det_interval: float = float(os.getenv("DET_INTERVAL", "0.5"))  # seconds between YOLO frames
    window_len: float = float(os.getenv("WINDOW_LEN", "2.5"))
    motion_dead_gate: float = float(os.getenv("MOTION_DEAD_GATE", "2.0"))

    # Commentary
    min_gap_for_filler: float = float(os.getenv("MIN_GAP_FOR_FILLER", "25.0"))
    max_commentary_words: int = int(os.getenv("MAX_COMMENTARY_WORDS", "18"))
    gpt2_lora_dir: Path = Path(os.getenv(
        "GPT2_LORA_DIR",
        str(_MODELS_DIR / ("gpt2_lora_v3"
                           if (_MODELS_DIR / "gpt2_lora_v3").exists()
                           else "gpt2_lora_v2"
                           if (_MODELS_DIR / "gpt2_lora_v2").exists()
                           else "gpt2_lora"))))

    # Devices
    device: str = os.getenv("DEVICE", "0" if os.getenv("FORCE_CPU") != "1" else "cpu")


settings = Settings()

YOLO_WEIGHTS = settings.models_dir / "yolov8_football" / "run1" / "weights" / "best.pt"
YOLO_FALLBACK = settings.models_dir / "yolov8n.pt"
RESNET_WEIGHTS = settings.models_dir / "resnet18_events.pt"

# Score-bug goal anchors for the CURRENT job, set by the pipeline at runtime
# (list of {"t": sec_absolute, "side": "home"|"away", "minute": int}).
# Empty = no bug evidence; heuristics run ungated as before.
SCORE_ANCHORS: list[dict] = []

EVENT_CLASSES = ["goal", "shot", "pass", "tackle", "dead_time"]
