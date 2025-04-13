import os
import time

import torch
import torchaudio
from einops import rearrange
from pydub import AudioSegment

from infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)

def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    chunked=False,
    steps=32,
    cfg_strength=4
):
    with torch.inference_mode():
        generated, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=steps,
            cfg_strength=cfg_strength,
            start_time=start_time,
        )

        generated = generated.to(torch.float32)
        latent = generated.transpose(1, 2)  # [b d t]

        output = decode_audio(latent, vae_model, chunked=chunked)

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        # Peak normalize, clip, convert to int16, and save to file
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        return output

def process(
    lrc_path=None, # lyrics of target song
    ref_prompt=None, # reference prompt as style prompt for target song
    ref_audio_path=None, # reference audio as style prompt for target song
    chunked=False, # whether to use chunked decoding
    audio_length=95, # length of generated song
    repo_id="ASLP-lab/DiffRhythm-base", # target model
    output_dir="infer/example/output", # output directory of target song
    lyrics_content=None,
    steps=32,
    cfg_strength=4
):
    print("Current working directory:", os.getcwd())

    print("lrc_path:", lrc_path)
    print("ref_prompt:", ref_prompt)
    print("ref_audio_path:", ref_audio_path) 
    print("chunked:", chunked)
    print("audio_length:", audio_length)
    print("repo_id:", repo_id)
    print("output_dir:", output_dir)
    print("lyrics_content:", lyrics_content)
    print("steps:", steps)
    print("cfg_strength:", cfg_strength)
    
    assert (
        ref_prompt or ref_audio_path
    ), "either ref_prompt or ref_audio_path should be provided"
    
    assert not (
        ref_prompt and ref_audio_path
    ), "only one of them should be provided"
    
    assert audio_length in ['95s', '285s'], "audio_length must be either 95s or 285s"
    audio_length = 95 if audio_length == '95s' else 285

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    # Set max frames based on audio length
    if audio_length == 95:
        max_frames = 2048
    elif audio_length == 285:
        max_frames = 6144

    # Load models
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device, repo_id=repo_id)

    # Process lyrics
    if lyrics_content:
        lrc = lyrics_content
    elif lrc_path:
        with open(lrc_path, "r", encoding='utf-8') as f:
            lrc = f.read()
    else:
        lrc = ''
    
    lrc_prompt, start_time = get_lrc_token(max_frames, lrc, tokenizer, device)

    # Get style prompt
    if ref_audio_path:
        style_prompt = get_style_prompt(muq, ref_audio_path)
    else:
        style_prompt = get_style_prompt(muq, prompt=ref_prompt)

    negative_style_prompt = get_negative_style_prompt(device)
    latent_prompt = get_reference_latent(device, max_frames)

    # Run inference
    s_t = time.time()
    generated_song = inference(
        cfm_model=cfm,
        vae_model=vae,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=max_frames,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        chunked=chunked,
        steps=steps,
        cfg_strength=cfg_strength
    )
    e_t = time.time() - s_t
    print(f"inference cost {e_t:.2f} seconds")

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    output_filename = f"output_{timestamp}"
    wav_path = os.path.join(output_dir, f"{output_filename}.wav")
    mp3_path = os.path.join(output_dir, f"{output_filename}.mp3")
    
    # Save WAV first
    torchaudio.save(wav_path, generated_song, sample_rate=44100)
    
    # Convert to MP3
    try:
        sound = AudioSegment.from_wav(wav_path)
        sound.export(mp3_path, format="mp3")
        print(f"MP3 file saved at: {mp3_path}")
        
        # Optionally remove WAV file if no longer needed
        # os.remove(wav_path)
        
        return mp3_path
    except ImportError:
        print("Warning: pydub not installed. Returning WAV file instead.")
        return wav_path
