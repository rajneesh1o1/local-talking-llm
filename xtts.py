import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

# PyTorch 2.6: allowlist XTTS config class for safe checkpoint loading
add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

speaker_wav = "/Users/rajneesh/Desktop/projects/local-talking-llm/voices/tinker_bell_trimmed.wav"

# wav = tts.tts(text="what the fuck went wrong here everything is messed up now!", speaker_wav=speaker_wav, language="en")
# tts.tts_to_file(text="what the fuck went wrong here everything is messed up now!", speaker_wav=speaker_wav, language="en", file_path="output.wav")

tts.tts_to_file(text="♪When the moon comes up to the shine of a face, the birds are fast asleep, and the lanterns hang from every post the fairies leave the keep, they join their hands and sing their songs, that nary a soul can hear, in the springtime when the earth is new, to the fairies they draw near, to the fairies they draw near♪", speaker_wav=speaker_wav, language="en", file_path="output.wav")

