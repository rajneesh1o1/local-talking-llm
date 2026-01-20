import warnings
import os
import sys
import re
import threading
import queue
import numpy as np
import sounddevice as sd
import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

# Suppress warnings - must be done before importing other modules
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress stderr for librosa warnings
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self._original_stderr

# PyTorch 2.6: allowlist XTTS config class for safe checkpoint loading
add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

# Get device - XTTS works better on CPU for Mac (MPS has poor support and is slower)
# Use CUDA if available (NVIDIA GPU), otherwise use CPU
if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA (NVIDIA GPU)")
else:
    device = "cpu"
    print("Using CPU (MPS disabled - XTTS is faster on CPU for Mac)")

print(f"Device: {device}")

# Init TTS (suppress warnings during initialization)
with SuppressStderr():
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Optimize PyTorch for faster inference
torch.set_num_threads(4)  # Use multiple CPU threads
torch.set_num_interop_threads(4)

speaker_wav = "/Users/rajneesh/Desktop/projects/local-talking-llm/voices/tinker_bell_trimmed.wav"

def split_text_into_chunks(text, max_chars=240, min_chars=3):
    """Split text into chunks, using ._. as a custom separator, then sentence boundaries."""
    text = text.strip()
    
    # First, check if ._. exists in text - if so, split on it first
    if "._." in text:
        # Split on ._. separator (this is a custom chunking marker, not spoken)
        parts = text.split("._.")
        chunks = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Remove any remaining ._. markers and clean up
            part = part.replace("._.", "").strip()
            if part:
                chunks.append(part)
        
        # If we got chunks from ._. splitting, return them (they're already separated)
        if chunks:
            return chunks
    
    # If no ._. separator, use normal chunking logic
    # If text is very short, just return it as a single chunk
    if len(text) <= max_chars:
        # Only filter if it's truly empty or just punctuation/musical notes
        if text and not re.match(r'^[♪\s.!?]+$', text):
            return [text]
        elif text:  # Even if it's just punctuation, return it if it's the whole text
            return [text]
        else:
            return []
    
    # Split by sentence endings first, then by commas if needed
    # Don't split on musical notes - treat them as part of the text
    sentences = re.split(r'([.!?]\s+)', text)
    
    chunks = []
    current_chunk = ""
    
    for part in sentences:
        if not part.strip():
            continue
        
        part = part.strip()
        potential_chunk = current_chunk + " " + part if current_chunk else part
        
        # If adding this part would exceed max_chars
        if len(potential_chunk) > max_chars:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If the part itself is too long, split it by commas
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
    
    # Add remaining chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Merge very small chunks with adjacent chunks (but only if we have multiple chunks)
    if len(chunks) > 1:
        merged_chunks = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            # If chunk is too small and not the last one, merge with next
            if len(chunk) < min_chars and i < len(chunks) - 1:
                merged_chunks.append(chunk + " " + chunks[i + 1])
                i += 2
            else:
                merged_chunks.append(chunk)
                i += 1
        chunks = merged_chunks
    
    # Filter out chunks that are ONLY punctuation or musical notes (but keep them if mixed with text)
    filtered_chunks = []
    for chunk in chunks:
        chunk_stripped = chunk.strip()
        # Keep chunk if it has actual text content (not just punctuation/musical notes)
        if chunk_stripped and not re.match(r'^[♪\s.!?]+$', chunk_stripped):
            filtered_chunks.append(chunk)
    
    # If all chunks were filtered but we had chunks, return the original chunks
    # (this handles edge cases where text might be mostly punctuation)
    return filtered_chunks if filtered_chunks else chunks if chunks else [text] if text else []

def play_audio_queue(audio_queue, sample_rate=22050):
    """Play audio chunks from queue."""
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:  # Sentinel value to stop
            break
        sd.play(audio_chunk, sample_rate)
        sd.wait()  # Wait until playback is finished
        audio_queue.task_done()

def tts_with_live_playback(text, speaker_wav, language="en", save_file=None, max_chars=240):
    """Generate TTS audio and play it live as chunks are generated."""
    # Split text into chunks (larger chunks = fewer generations = faster)
    chunks = split_text_into_chunks(text, max_chars=max_chars)
    
    # Clean up chunks: remove any ._. markers that might remain (they're separators, not spoken)
    chunks = [chunk.replace("._.", "").strip() for chunk in chunks if chunk.strip()]
    
    print(f"Split text into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk[:60]}...")
    
    # Create queue for audio chunks
    audio_queue = queue.Queue()
    
    # Start playback thread
    playback_thread = threading.Thread(
        target=play_audio_queue,
        args=(audio_queue,),
        daemon=True
    )
    playback_thread.start()
    
    # Check if we have any chunks
    if not chunks:
        print("Warning: No chunks to generate. Text might be empty or filtered out.")
        return
    
    # Generate and queue audio chunks
    all_audio = []
    for i, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():
            print(f"Skipping empty chunk {i+1}")
            continue
        # Final cleanup: ensure no ._. markers in the chunk (they're separators, not spoken)
        chunk = chunk.replace("._.", "").strip()
        if not chunk:
            print(f"Skipping empty chunk {i+1}")
            continue
            
        print(f"Generating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        try:
            # Use inference mode for faster execution (disables gradient computation)
            with torch.inference_mode(), SuppressStderr():
                wav = tts.tts(text=chunk, speaker_wav=speaker_wav, language=language)
        except Exception as e:
            print(f"Error generating chunk {i+1}: {e}")
            continue
        
        # Convert to numpy array - handle list, torch.Tensor, or numpy array
        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)
        elif isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        elif not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        
        # Ensure mono audio (1D array)
        if len(wav.shape) > 1:
            wav = wav[0] if wav.shape[0] == 1 else np.mean(wav, axis=0)
        
        # Ensure it's a 1D array
        wav = wav.flatten()
        
        # Skip empty or very short audio chunks (likely noise)
        if len(wav) < 100:  # Less than ~5ms at 22050 Hz
            print(f"  Skipping very short chunk ({len(wav)} samples)")
            continue
        
        # Normalize audio to prevent clipping
        max_val = np.abs(wav).max()
        if max_val > 0:
            wav = wav / max_val * 0.8  # Scale to 80% to avoid clipping
        
        # Queue for playback (will start playing immediately if queue is empty)
        audio_queue.put(wav)
        
        # Store for saving
        all_audio.append(wav)
    
    # Wait for all audio to finish playing
    audio_queue.join()
    
    # Stop playback thread
    audio_queue.put(None)
    playback_thread.join()
    
    # Save combined audio if requested
    if save_file and all_audio:
        combined_audio = np.concatenate(all_audio)
        import soundfile as sf
        sf.write(save_file, combined_audio, 22050)
        print(f"Saved audio to {save_file}")
    elif save_file:
        print(f"Warning: No audio generated to save to {save_file}")

# Example usage
text = "When the moon comes up to the shine of a face,._. the birds are fast asleep, and the lanterns hang ._. from every post the fairies leave the keep, ._. they join their hands and sing their songs, ._. that nary a soul can hear, in the springtime when the earth is new, ._. to the fairies they draw near, to the fairies they draw near"



tts_with_live_playback(
    text="hi there",
    speaker_wav=speaker_wav,
    language="en",
    save_file="hi.wav"  # Optional: still save the file
)

tts_with_live_playback(
    text=text,
    speaker_wav=speaker_wav,
    language="en",
)


