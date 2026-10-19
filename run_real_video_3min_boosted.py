"""Run the pipeline on the 3-minute clip with boosted commentary volume."""

import os
import sys
sys.path.insert(0, r'C:\Users\asus\football-commentator')

from app.pipeline import run_pipeline
from pathlib import Path

# Use the downloaded match video
video_path = r'C:\Users\asus\Downloads/3 - 3 ARG vs FRA - Second-Half - Bein.mp4'
# Process first 3 minutes (180 seconds)
t0 = 0.0
duration = 180.0  # 3 minutes
# Output directory
out_dir = r'C:\Users\asus\football-commentator/output_match_3min_boosted'

print(f"Testing pipeline on {video_path} from {t0} to {t0+duration} seconds with boosted commentary volume...")
print(f"Output will be written to {out_dir}")

# Set environment variable to boost the commentary voice
os.environ['VOICE_BOOST'] = '2.0'  # Double the volume

# Progress reporter
def dummy_progress(stage, frac, msg=""):
    print(f"[{stage}] {frac*100:.0f}%: {msg}")

def dummy_on_transcript(lines):
    # Optionally print transcript lines
    for line in lines:
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