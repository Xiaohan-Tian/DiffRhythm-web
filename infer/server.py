import gradio as gr
import threading
import time
from process import process
import os
import random
import argparse
from datetime import datetime
from pydub import AudioSegment

def save_uploaded_audio(file_obj):
    # Create upload directory if it doesn't exist
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example", "upload")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Get file extension and timestamp
    original_path = file_obj.name if file_obj and hasattr(file_obj, 'name') else file_obj # Gradio backwards compatibility
    file_ext = os.path.splitext(original_path)[1].lower() if original_path else None
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Save with new filename
    base_filename = f"ref_{timestamp}"
    
    if file_ext == ".mp3":
        mp3_path = os.path.join(upload_dir, f"{base_filename}.mp3")
        wav_path = os.path.join(upload_dir, f"{base_filename}.wav")
        
        # First save the MP3 file
        with open(mp3_path, 'wb') as f:
            with open(original_path, 'rb') as orig:
                f.write(orig.read())
            
        # Convert to WAV
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav")
        
        return wav_path
        
    elif file_ext == ".wav":
        wav_path = os.path.join(upload_dir, f"{base_filename}.wav")
        
        # Save the WAV file
        with open(wav_path, 'wb') as f:
            with open(original_path, 'rb') as orig:
                f.write(orig.read())
            
        return wav_path
    
    else:
        raise ValueError("Unsupported file format. Please upload a .wav or .mp3 file.")

def run_generation(
    genre_prompt, 
    lyrics, 
    length,
    ref_audio=None,
    chunked_decoding=True,
    steps=32,
    cfg_strength=4
):
    try:        
        if ref_audio:
            # Use ref_audio instead of genre_prompt
            output_path = process(
                ref_prompt=None,
                ref_audio_path=ref_audio,
                lyrics_content=lyrics, 
                chunked=chunked_decoding, 
                audio_length=length, 
                repo_id="ASLP-lab/DiffRhythm-base" if length == '95s' else "ASLP-lab/DiffRhythm-full", 
                steps=steps,
                cfg_strength=cfg_strength
            )
        else:
            # Use genre_prompt
            output_path = process(
                ref_prompt=genre_prompt, 
                lyrics_content=lyrics, 
                chunked=chunked_decoding, 
                audio_length=length, 
                repo_id="ASLP-lab/DiffRhythm-base" if length == '95s' else "ASLP-lab/DiffRhythm-full", 
                steps=steps,
                cfg_strength=cfg_strength
            )
        return "Generation complete!", output_path
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)  # Print to console
        return error_msg, None  # Return to Gradio UI

# Load default content from files
def load_text_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return ""

# Get the absolute path to the files
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
lyrics_path = os.path.join(base_dir, "infer/example", "eg_en.lrc")

# Get default values from prompt_egs files
genre_default = "classical genres, hopeful mood, piano."
lyrics_default = load_text_file(lyrics_path)

# Create the Gradio interface
with gr.Blocks() as demo:
    # Title area
    gr.Markdown("# DiffRhythm Gradio GUI")
    gr.Markdown("""
    Official Website: [DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)
    """)
    
    # Two-column layout
    with gr.Row():
        # Left column
        with gr.Column(scale=1):
            genre_prompt = gr.Textbox(
                label="Genres Prompt",
                placeholder="classical genres, hopeful mood, piano.",
                lines=3,
                value=genre_default
            )
            
            lyrics = gr.Textbox(
                label="Lyrics",
                placeholder="",
                lines=30,
                value=lyrics_default
            )
        
        # Right column
        with gr.Column(scale=1):
            chunked_decoding = gr.Checkbox(
                label="Chunked Decoding (Check this if you have <= 8GB of VRAM)",
                value=True
            )

            # Replace the dropdown with radio buttons
            length_radio = gr.Radio(
                choices=["95s", "285s"],
                value="95s",
                label="Music Duration",
                interactive=True
            )

            steps_slider = gr.Slider(
                minimum=1,
                maximum=64,
                value=32,
                label="Steps",
                interactive=True,
                step=1  # Add step=1 to only allow integer values
            )

            cfg_strength_slider = gr.Slider(
                minimum=0.0,
                maximum=10.0,
                value=4.0,
                label="CFG Strength",
                interactive=True
            )

            ref_audio = gr.Audio(
                label="Reference Audio (optional)",
                type="filepath",
                interactive=True
            )
            
            status = gr.Textbox(
                label="Status", 
                interactive=False
            )
            
            generate_button = gr.Button("Generate")
            
            # Last Generated Song (audio player)
            audio_output = gr.Audio(
                label="Last Generated Song", 
                interactive=False,
                type="filepath"
            )
    
    # Event handlers
    ref_audio.upload(
        fn=save_uploaded_audio,
        inputs=[ref_audio],
        outputs=[ref_audio]
    )
    
    generate_button.click(
        fn=run_generation,
        inputs=[genre_prompt, lyrics, length_radio, ref_audio, chunked_decoding, steps_slider, cfg_strength_slider],
        outputs=[status, audio_output]
    )

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch DiffRhythm Gradio GUI")
    parser.add_argument("--share", action="store_true", default=False, 
                        help="Whether to create a shareable link (default: True)")
    parser.add_argument("--port", type=int, default=None, 
                        help="Port to run the server on (default: Gradio default)")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                        help="Host to bind to (default: 127.0.0.1)")
    
    args = parser.parse_args()
    
    # Launch the interface with the specified parameters
    demo.queue().launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port
    ) 