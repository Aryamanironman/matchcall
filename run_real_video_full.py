"""Run the pipeline on the full match video."""

import os
import sys
sys.path.insert(0, r'C:\Users\asus\football-commentator')

from app.pipeline import run_pipeline
from pathlib import Path

# Use the downloaded match video
video_path = r'C:\Users\asus\Downloads/3 - 3 ARG vs FRA - Second-Half - Bein.mp4'
# Process the entire video (duration=None means until end)
t0 = 0.0
duration = None  # Process until the end of the video
# Output directory for full video run
out_dir = r'C:\Users\asus\football-commentator/output_match_full'

print(f"Testing pipeline on {video_path} from start to end...")
print(f"Output will be written to {out_dir}")

# Progress reporter
def dummy_progress(stage, frac, msg=""):
    print(f"[{stage}] {frac*100:.0f}%: {msg}")

def dummy_on_transcript(lines):
    # Optionally print transcript lines (first few and last few)
    if len(lines) <= 10:
        for line in lines:
            print(f"TRANSCRIPT: {line}")
    else:
        for line in lines[:5]:
            print(f"TRANSCRIPT: {line}")
        print("...")
        for line in lines[-5:]:
            print(f"TRANSCRIPT: {line}")

def dummy_on_segment(seg):
    pass

def dummy_cancel_check():
    return False

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

try:
    result = run_pipeline(
        video=video_path,
        home="ARG",
        away="FRA",
        t0=t0,
        duration=duration,
        out_dir=Path(out_dir),
        progress=dummy_progress,
        on_transcript=dummy_on_transcript,
        on_segment=dummy_on_segment,
        _cancel_check=dummy_cancel_check,
        orig_audio=True  # use original audio
    )
    print("Pipeline completed successfully!")
    print(f"Result: {result}")
except Exception as e:
    print(f"Pipeline failed with error: {e}")
    import traceback
    traceback.print_exc()