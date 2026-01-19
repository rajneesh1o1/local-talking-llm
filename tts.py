import nltk
import torch
import warnings
import numpy as np
import torchaudio as ta
import types
import torch.nn.functional as F
from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.s3tokenizer import drop_invalid_tokens

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    def __init__(self, device: str | None = None):
        """
        Initializes the TextToSpeechService class with ChatterBox TTS.

        Args:
            device (str, optional): The device to be used for the model. If None, will auto-detect.
                Can be "cuda", "mps", or "cpu".
        """
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
        """Ensure required NLTK tokenizer data is present for sent_tokenize."""
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            nltk.download("punkt_tab")

    def _patch_chatterbox_generate(self):
        """
        Monkeypatch ChatterboxTTS.generate to accept `max_new_tokens` (the upstream code hardcodes 1000).
        This lets us reduce sampling steps and speed up TTS substantially without editing site-packages.
        """

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

            # Update exaggeration if needed
            if exaggeration != model_self.conds.t3.emotion_adv[0, 0, 0]:
                _cond = model_self.conds.t3
                from chatterbox.models.t3.modules.cond_enc import T3Cond

                model_self.conds.t3 = T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=exaggeration * torch.ones(1, 1, 1),
                ).to(device=model_self.device)

            # Norm and tokenize text
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
                # Extract only the conditional batch.
                speech_tokens = speech_tokens[0]
                speech_tokens = drop_invalid_tokens(speech_tokens).to(model_self.device)

                wav, _ = model_self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=model_self.conds.gen,
                )
                wav = wav.squeeze(0).detach().cpu().numpy()
                watermarked_wav = model_self.watermarker.apply_watermark(wav, sample_rate=model_self.sr)
            return torch.from_numpy(watermarked_wav).unsqueeze(0)

        # Patch instance method
        self.model.generate = types.MethodType(generate_patched, self.model)

    def _patch_torch_load(self):
        """
        Patches torch.load to automatically map tensors to the correct device.
        This is needed because ChatterBox models may have been saved on CUDA.
        """
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
        """
        Synthesizes audio from the given text using ChatterBox TTS.

        Args:
            text (str): The input text to be synthesized.
            audio_prompt_path (str, optional): Path to audio file for voice cloning. Defaults to None.
            exaggeration (float, optional): Emotion exaggeration control (0-1). Defaults to 0.5.
            cfg_weight (float, optional): Control for pacing and delivery. Defaults to 0.5.

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        # Cache conditioning so we don't re-embed the same reference audio every turn.
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

        # Convert tensor to numpy array format compatible with sounddevice
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
        """
        Synthesizes audio from the given long-form text using ChatterBox TTS.

        Args:
            text (str): The input text to be synthesized.
            audio_prompt_path (str, optional): Path to audio file for voice cloning. Defaults to None.
            exaggeration (float, optional): Emotion exaggeration control (0-1). Defaults to 0.5.
            cfg_weight (float, optional): Control for pacing and delivery. Defaults to 0.5.

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
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
        """
        Saves a voice sample to file for later use as voice prompt.

        Args:
            text (str): The text to synthesize.
            output_path (str): Path where to save the audio file.
            audio_prompt_path (str, optional): Path to audio file for voice cloning.
        """
        if audio_prompt_path:
            if (self._last_audio_prompt_path != audio_prompt_path) or (self.model.conds is None):
                self.model.prepare_conditionals(audio_prompt_path)
                self._last_audio_prompt_path = audio_prompt_path
            audio_prompt_path = None

        wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)
        ta.save(output_path, wav, self.sample_rate)
