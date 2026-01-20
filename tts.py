import nltk
import torch
import warnings
import numpy as np
import torchaudio as ta
import types
import torch.nn.functional as F
import subprocess
import tempfile
import os
from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.s3tokenizer import drop_invalid_tokens

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    def __init__(self, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        self._ensure_nltk()
        self._patch_torch_load()
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sample_rate = self.model.sr
        self._last_audio_prompt_path: str | None = None
        self._patch_chatterbox_generate()

    def _ensure_nltk(self):
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            nltk.download("punkt_tab")

    def _patch_chatterbox_generate(self):
        # Monkeypatch: allow max_new_tokens (upstream hardcodes 1000)

        def generate_patched(
            model_self,
            text,
            audio_prompt_path=None,
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.8,
            max_new_tokens=300,
        ):
            if audio_prompt_path:
                model_self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            else:
                assert model_self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

            if exaggeration != model_self.conds.t3.emotion_adv[0, 0, 0]:
                _cond = model_self.conds.t3
                from chatterbox.models.t3.modules.cond_enc import T3Cond

                model_self.conds.t3 = T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=exaggeration * torch.ones(1, 1, 1),
                ).to(device=model_self.device)

            text = punc_norm(text)
            text_tokens = model_self.tokenizer.text_to_tokens(text).to(model_self.device)
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

            sot = model_self.t3.hp.start_text_token
            eot = model_self.t3.hp.stop_text_token
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)

            with torch.inference_mode():
                speech_tokens = model_self.t3.inference(
                    t3_cond=model_self.conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                )
                speech_tokens = speech_tokens[0]
                speech_tokens = drop_invalid_tokens(speech_tokens).to(model_self.device)

                wav, _ = model_self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=model_self.conds.gen,
                )
                wav = wav.squeeze(0).detach().cpu().numpy()
                watermarked_wav = model_self.watermarker.apply_watermark(wav, sample_rate=model_self.sr)
            return torch.from_numpy(watermarked_wav).unsqueeze(0)

        self.model.generate = types.MethodType(generate_patched, self.model)

    def _patch_torch_load(self):
        map_location = torch.device(self.device)

        if not hasattr(torch, '_original_load'):
            torch._original_load = torch.load

        def patched_torch_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = map_location
            return torch._original_load(*args, **kwargs)

        torch.load = patched_torch_load

    def synthesize(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        max_new_tokens: int = 300,
    ):
        # Cache conditioning for repeated voice prompts.
        if audio_prompt_path:
            if (self._last_audio_prompt_path != audio_prompt_path) or (self.model.conds is None):
                self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
                self._last_audio_prompt_path = audio_prompt_path
            audio_prompt_path = None  # already prepared

        wav = self.model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        audio_array = wav.squeeze().cpu().numpy()
        return self.sample_rate, audio_array

    def long_form_synthesize(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.7,
        cfg_weight: float = 0.7,
        temperature: float = 0.9,
        max_new_tokens: int = 1000,
    ):
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(
                sent,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            pieces += [audio_array, silence.copy()]

        return self.sample_rate, np.concatenate(pieces)

    def save_voice_sample(self, text: str, output_path: str, audio_prompt_path: str | None = None):
        if audio_prompt_path:
            if (self._last_audio_prompt_path != audio_prompt_path) or (self.model.conds is None):
                self.model.prepare_conditionals(audio_prompt_path)
                self._last_audio_prompt_path = audio_prompt_path
            audio_prompt_path = None

        wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)
        ta.save(output_path, wav, self.sample_rate)


class PiperTTSService:
    """
    Piper TTS backend (fast). Uses the `piper` CLI from `piper-tts`.
    Requires a voice model `.onnx` file (and typically a `.onnx.json` config).
    """

    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        speaker: int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w

    def _resolve_config(self) -> str | None:
        if self.config_path and os.path.exists(self.config_path):
            return self.config_path

        # Common Piper convention: model.onnx + model.onnx.json
        candidate = f"{self.model_path}.json"
        if os.path.exists(candidate):
            return candidate

        # Fallback: pick a .onnx.json from the same directory (handles casing mismatches)
        model_dir = os.path.dirname(self.model_path) or "."
        try:
            configs = [f for f in os.listdir(model_dir) if f.lower().endswith(".onnx.json")]
        except FileNotFoundError:
            return None

        if not configs:
            return None

        model_name = os.path.basename(self.model_path).lower()
        for cfg in configs:
            if cfg.lower().startswith(model_name):
                return os.path.join(model_dir, cfg)

        if len(configs) == 1:
            return os.path.join(model_dir, configs[0])

        return os.path.join(model_dir, configs[0])

    def synthesize(
        self,
        text: str,
        audio_prompt_path: str | None = None,  # unused; kept for interface compatibility
        exaggeration: float = 0.5,  # unused
        cfg_weight: float = 0.5,  # unused
        temperature: float = 0.8,  # unused
        max_new_tokens: int = 0,  # unused
    ):
        import soundfile as sf

        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Piper model not found: {self.model_path}")

        cfg = self._resolve_config()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        cmd = ["piper", "--model", self.model_path, "--output_file", out_path]
        if cfg:
            cmd += ["--config", cfg]
        if self.speaker is not None:
            cmd += ["--speaker", str(self.speaker)]
        if self.length_scale is not None:
            cmd += ["--length_scale", str(self.length_scale)]
        if self.noise_scale is not None:
            cmd += ["--noise_scale", str(self.noise_scale)]
        if self.noise_w is not None:
            cmd += ["--noise_w", str(self.noise_w)]

        try:
            subprocess.run(cmd, input=text, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            audio, sr = sf.read(out_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            return int(sr), audio
        except FileNotFoundError as e:
            raise RuntimeError("Piper CLI not found. Install with: pip install piper-tts") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Piper failed: {e.stderr}") from e
        finally:
            try:
                os.remove(out_path)
            except Exception:
                pass

    def save_voice_sample(self, text: str, output_path: str, audio_prompt_path: str | None = None):
        sr, audio = self.synthesize(text)
        import soundfile as sf

        sf.write(output_path, audio, sr)
