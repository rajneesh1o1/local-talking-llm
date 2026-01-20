import warnings
import os
import sys
import re
import threading
import queue
import numpy as np
import sounddevice as sd
import torch
import librosa
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self._original_stderr

add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA (NVIDIA GPU)")
else:
    device = "cpu"
    print("Using CPU (MPS disabled - XTTS is faster on CPU for Mac)")

print(f"Device: {device}")

with SuppressStderr():
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

speaker_wav = "/Users/rajneesh/Desktop/projects/local-talking-llm/voices/tinker_bell_trimmed.wav"

def split_text_into_chunks(text, max_chars=240, min_chars=3):
    text = text.strip()
    
    if "#" in text:
        parts = text.split("#")
        chunks = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            part = part.replace("#", "").strip()
            if part:
                chunks.append(part)
        
        if chunks:
            return chunks
    
    if len(text) <= max_chars:
        if text and not re.match(r'^[♪\s.!?]+$', text):
            return [text]
        elif text:
            return [text]
        else:
            return []
    
    sentences = re.split(r'([.!?]\s+)', text)
    
    chunks = []
    current_chunk = ""
    
    for part in sentences:
        if not part.strip():
            continue
        
        part = part.strip()
        potential_chunk = current_chunk + " " + part if current_chunk else part
        
        if len(potential_chunk) > max_chars:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            if len(part) > max_chars:
                comma_parts = re.split(r'(,\s+)', part)
                temp_chunk = ""
                for comma_part in comma_parts:
                    if not comma_part.strip():
                        continue
                    if temp_chunk and len(temp_chunk) + len(comma_part) > max_chars:
                        if temp_chunk.strip():
                            chunks.append(temp_chunk.strip())
                        temp_chunk = comma_part.strip()
                    else:
                        temp_chunk += comma_part if temp_chunk.endswith(" ") or not temp_chunk else " " + comma_part
                current_chunk = temp_chunk.strip() if temp_chunk.strip() else part
            else:
                current_chunk = part
        else:
            current_chunk = potential_chunk
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    if len(chunks) > 1:
        merged_chunks = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if len(chunk) < min_chars and i < len(chunks) - 1:
                merged_chunks.append(chunk + " " + chunks[i + 1])
                i += 2
            else:
                merged_chunks.append(chunk)
                i += 1
        chunks = merged_chunks
    
    filtered_chunks = []
    for chunk in chunks:
        chunk_stripped = chunk.strip()
        if chunk_stripped and not re.match(r'^[♪\s.!?]+$', chunk_stripped):
            filtered_chunks.append(chunk)
    
    return filtered_chunks if filtered_chunks else chunks if chunks else [text] if text else []

def play_audio_queue(audio_queue, sample_rate=22050):
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break
        sd.play(audio_chunk, sample_rate)
        sd.wait()
        audio_queue.task_done()

# Generate TTS audio and play it live as chunks are generated. Use # in text to control chunk boundaries.
# speed: 1.0 = normal, >1.0 = faster, <1.0 = slower (e.g., 1.2 = 20% faster, 0.8 = 20% slower)
def tts_with_live_playback(text, speaker_wav, language="en", save_file=None, max_chars=240, speed=1.0):
    chunks = split_text_into_chunks(text, max_chars=max_chars)
    chunks = [chunk.replace("#", "").strip() for chunk in chunks if chunk.strip()]
    
    print(f"Split text into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk[:60]}...")
    
    audio_queue = queue.Queue()
    
    playback_thread = threading.Thread(
        target=play_audio_queue,
        args=(audio_queue,),
        daemon=True
    )
    playback_thread.start()
    
    if not chunks:
        print("Warning: No chunks to generate. Text might be empty or filtered out.")
        return
    
    all_audio = []
    for i, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():
            print(f"Skipping empty chunk {i+1}")
            continue
        chunk = chunk.replace("#", "").strip()
        if not chunk:
            print(f"Skipping empty chunk {i+1}")
            continue
            
        print(f"Generating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        try:
            with torch.inference_mode(), SuppressStderr():
                wav = tts.tts(text=chunk, speaker_wav=speaker_wav, language=language)
        except Exception as e:
            print(f"Error generating chunk {i+1}: {e}")
            continue
        
        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)
        elif isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        elif not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        
        if len(wav.shape) > 1:
            wav = wav[0] if wav.shape[0] == 1 else np.mean(wav, axis=0)
        
        wav = wav.flatten()
        
        if len(wav) < 100:
            print(f"  Skipping very short chunk ({len(wav)} samples)")
            continue
        
        if speed != 1.0:
            wav = librosa.effects.time_stretch(wav, rate=speed)
        
        max_val = np.abs(wav).max()
        if max_val > 0:
            wav = wav / max_val * 0.8
        
        audio_queue.put(wav)
        all_audio.append(wav)
    
    audio_queue.join()
    audio_queue.put(None)
    playback_thread.join()
    
    if save_file and all_audio:
        combined_audio = np.concatenate(all_audio)
        import soundfile as sf
        sf.write(save_file, combined_audio, 22050)
        print(f"Saved audio to {save_file}")
    elif save_file:
        print(f"Warning: No audio generated to save to {save_file}")

text = "When the moon comes up to the shine of a face,# the birds are fast asleep, and the lanterns hang # from every post the fairies leave the keep, # they join their hands and sing their songs, # that nary a soul can hear, in the springtime when the earth is new, # to the fairies they draw near, to the fairies they draw near"



tts_with_live_playback(
    text="hi there",
    speaker_wav=speaker_wav,
    language="en",
)

