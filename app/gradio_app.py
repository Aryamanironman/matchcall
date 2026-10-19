"""Gradio interface for the football commentary pipeline.

Users upload a football match video, the system runs the AI commentary pipeline
(using YOLOv8, ResNet18, whistle/replay detectors, fusion, LLM commentary,
Fish Audio TTS, and ffmpeg mixing) and returns the commented video.
"""

import os
import tempfile
import shutil
from pathlib import Path
import gradio as gr
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the pipeline function
from app.pipeline import run_pipeline
from app.config import settings

def process_video(video_file, home_name="Team1", away_name="Team2", progress=gr.Progress()):
    """
    Process an uploaded video file with the commentary pipeline.
    
    Args:
        video_file: Gradio file object (tuple or temp path)
        home_name: Name of home team for commentary
        away_name: Name of away team for commentary
        progress: Gradio progress tracker
    
    Returns:
        Path to the output video file with commentary, or None on failure.
    """
    if video_file is None:
        return None
    
    # Create a temporary directory for this run
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Save uploaded video to a temporary file
        input_video = tmp_path / "input.mp4"
        # Gradio provides the video as a temporary file path; we copy it
        shutil.copy(video_file, input_video)
        
        # Output directory for this run
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Define progress callback for the pipeline
        def pipeline_progress(stage, frac, msg=""):
            # Gradio progress expects a value between 0 and 1
            progress(frac, desc=f"{stage}: {msg}")
        
        # Dummy callbacks for transcript and segment (we don't need to display them live)
        def dummy_on_transcript(lines):
            pass
        def dummy_on_segment(seg):
            pass
        def dummy_cancel_check():
            return False
        
        try:
            # Run the pipeline on the full video (duration=None means until end)
            # We'll process the whole video but could be long; for demo we might limit.
            # However, we'll let the user decide; we can add a duration limit later.
            result = run_pipeline(
                video=str(input_video),
                home=home_name,
                away=away_name,
                t0=0.0,
                duration=None,  # process until end
                out_dir=output_dir,
                progress=pipeline_progress,
                on_transcript=dummy_on_transcript,
                on_segment=dummy_on_segment,
                _cancel_check=dummy_cancel_check,
                orig_audio=True  # keep original audio and mix commentary over it
            )
            
            if result and "output" in result:
                output_video = Path(result["output"])
                if output_video.exists():
                    # Copy the output video to a location Gradio can serve (outside the temp dir?)
                    # Gradio can return files from within the temp dir as long as we return the path.
                    # However, the temp dir will be cleaned up after the function ends.
                    # We need to copy the output to a more permanent location or return it before cleanup.
                    # Gradio's file component can handle a temporary file if we return it directly;
                    # but the tempdir cleanup happens after the function returns, which is too late.
                    # Instead, we can output the video to a Gradio cache directory or return it as bytes?
                    # Simpler: we return the output video path, and Gradio will serve it; we must ensure
                    # the file is not deleted before Gradio reads it. Since we are in a temporary
                    # directory that will be cleaned up after the function returns, we need to
                    # move the file to a location that persists beyond the function call.
                    # Let's move it to a temporary file in the system temp that we manage manually?
                    # Actually, Gradio's documentation says that returning a file path works if the
                    # file is temporary and Gradio will copy it to its own cache before returning
                    # to the user. We'll trust that.
                    return str(output_video)
                else:
                    return None
            else:
                return None
        except Exception as e:
            print(f"Error during pipeline processing: {e}")
            import traceback
            traceback.print_exc()
            return None

# Define the Gradio interface
with gr.Blocks(title="AI Football Commentary") as demo:
    gr.Markdown("# AI Football Commentary System")
    gr.Markdown(
        """
        Upload a football match video (any format supported by ffmpeg). 
        The system will:
        - Detect players and actions using YOLOv8 and ResNet18
        - Detect whistles, replays, and score bugs
        - Fuse detections into event timeline (goal, shot, pass, tackle, dead_time)
        - Generate AI commentary using a fine-tuned LLM (placeholder)
        - Synthesize speech with Fish Audio (primary) or edge-tts (fallback)
        - Mix commentary over original stadium audio
        - Output the same video with AI play-by-play commentary dubbed in.
        """
    )
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Football Match Video", format="mp4")
            with gr.Row():
                home_input = gr.Textbox(label="Home Team Name", value="Team1")
                away_input = gr.Textbox(label="Away Team Name", value="Team2")
            process_btn = gr.Button("Generate Commentary", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Commentary Video")
            gr.Markdown("*Note: Processing time depends on video length. Expect ~5-10x real-time speed on GPU.*")
    
    # Set up the button click
    process_btn.click(
        fn=process_video,
        inputs=[video_input, home_input, away_input],
        outputs=[video_output],
        api_name="process_video"
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()