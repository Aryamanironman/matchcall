"""Test the Gradio processing function with a short video clip."""

import os
import sys
sys.path.insert(0, r'C:\Users\asus\football-commentator')

from app.gradio_app import process_video
from pathlib import Path

# Use the synthetic match video (short, 5 seconds) for quick test
video_path = r'C:\Users\asus\football-commentator/data/test/synthetic_match.mp4'
if not Path(video_path).exists():
    # Fallback to the first 5 seconds of the real video? We'll just use the synthetic.
    print(f"Synthetic video not found at {video_path}")
    sys.exit(1)

print(f"Testing process_video on {video_path}...")
# We'll process the first 5 seconds only to keep it fast.
# However, the process_video function currently processes the whole video (duration=None).
# We can modify it to accept duration, but for now we'll just process the whole synthetic video (it's only 5 seconds).
output = process_video(video_path, home="Home", away="Away")
if output:
    print(f"SUCCESS: Output video created at {output}")
    # Check that the file exists and is not empty
    if Path(output).exists():
        size = Path(output).stat().st_size
        print(f"Output file size: {size} bytes")
        if size > 1000:  # at least 1KB
            print("Output looks valid.")
        else:
            print("Warning: output file is very small.")
    else:
        print("Error: output path does not exist.")
else:
    print("FAILED: process_video returned None.")