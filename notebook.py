"""
Colab (T4 GPU) quickstart:

Cell 1:
!pip -q install -U pip
!pip -q install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install openai-whisper soundfile numpy nltk
!pip -q install chatterbox-tts transformers accelerate
!pip -q install langchain langchain-ollama ollama

Cell 2:
!python notebook.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def _pick_device(requested: str | None) -> str:
    """Pick cuda/mps/cpu depending on availability, unless explicitly requested."""
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _auto_tts_steps(text: str, requested_steps: int) -> int:
    """Auto-scale TTS steps based on response length to avoid cut-off audio."""
    if requested_steps and requested_steps > 0:
        return requested_steps
    word_count = max(1, len(text.split()))
    # Tuned to be usable on consumer GPUs while avoiding truncation for short answers.
    return max(150, min(600, word_count * 20))


# ----------------------------
# TTS (Chatterbox) — bundled
# ----------------------------

class TextToSpeechService:
    """
    Chatterbox TTS wrapper with:
    - NLTK data auto-download for sentence tokenization
    - Conditioning cache (voice prompt embedding reused)
    - Monkeypatch to allow configurable max_new_tokens (upstream hardcodes 1000)
    """

    def __init__(self, device: str | None = None):
        import nltk
        import torch
        import warnings

        warnings.filterwarnings(
            "ignore",
            message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
        )

        self.device = _pick_device(device)
        print(f"Using device: {self.device}")

        # Ensure NLTK punkt_tab exists (NLTK 3.9+ uses punkt_tab for sent_tokenize)
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            nltk.download("punkt_tab")

        # Patch torch.load to map tensors to chosen device (avoids CUDA-saved checkpoint issues)
        self._patch_torch_load(torch)

        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sample_rate = self.model.sr

        self._last_audio_prompt_path: str | None = None
        self._patch_chatterbox_generate()

    def _patch_torch_load(self, torch_mod):
        map_location = torch_mod.device(self.device)
        if not hasattr(torch_mod, "_original_load"):
            torch_mod._original_load = torch_mod.load  # type: ignore[attr-defined]

        def patched_torch_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = map_location
            return torch_mod._original_load(*args, **kwargs)  # type: ignore[attr-defined]

        torch_mod.load = patched_torch_load  # type: ignore[assignment]

    def _patch_chatterbox_generate(self):
        import types
        import torch
        import torch.nn.functional as F
        from chatterbox.tts import punc_norm
        from chatterbox.models.s3tokenizer import drop_invalid_tokens
        from chatterbox.models.t3.modules.cond_enc import T3Cond

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
                model_self.conds.t3 = T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=exaggeration * torch.ones(1, 1, 1),
                ).to(device=model_self.device)

            # Norm + tokenize text
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
                speech_tokens = drop_invalid_tokens(speech_tokens[0]).to(model_self.device)
                wav, _ = model_self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=model_self.conds.gen,
                )
                wav = wav.squeeze(0).detach().cpu().numpy()
                watermarked_wav = model_self.watermarker.apply_watermark(wav, sample_rate=model_self.sr)
            return torch.from_numpy(watermarked_wav).unsqueeze(0)

        self.model.generate = types.MethodType(generate_patched, self.model)

    def synthesize(
        self,
        text: str,
        *,
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        max_new_tokens: int = 300,
    ) -> tuple[int, np.ndarray]:
        # Cache conditioning so we don't re-embed the same voice prompt each turn
        if audio_prompt_path:
            if (self._last_audio_prompt_path != audio_prompt_path) or (self.model.conds is None):
                self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
                self._last_audio_prompt_path = audio_prompt_path
            audio_prompt_path = None

        wav = self.model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return self.sample_rate, wav.squeeze().cpu().numpy()


# ----------------------------
# STT (Whisper) — optional
# ----------------------------

def transcribe_audio_file(audio_path: str, *, whisper_model: str, device: str) -> str:
    import whisper

    stt = whisper.load_model(whisper_model, device=device)
    # whisper transcribe accepts file path directly
    result = stt.transcribe(audio_path, fp16=(device == "cuda"))
    return (result.get("text") or "").strip()


# ----------------------------
# LLM — Ollama or HF fallback
# ----------------------------

@dataclass
class LLMConfig:
    use_ollama: bool
    ollama_url: str
    ollama_model: str
    hf_model: str | None
    max_new_tokens: int


def generate_llm_response(prompt: str, *, cfg: LLMConfig) -> str:
    if cfg.use_ollama:
        # Ollama via LangChain
        from langchain_ollama import OllamaLLM
        from langchain_core.prompts import ChatPromptTemplate

        llm = OllamaLLM(model=cfg.ollama_model, base_url=cfg.ollama_url.strip())
        tpl = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful and friendly assistant. Keep replies concise (1-2 sentences)."),
                ("human", "{input}"),
            ]
        )
        chain = tpl | llm
        out = chain.invoke({"input": prompt})
        return str(out).strip()

    # HF Transformers fallback (no Ollama required)
    if not cfg.hf_model:
        raise RuntimeError(
            "No Ollama configured and no --hf-model provided. "
            "Either pass --ollama-url/--model, or pass --hf-model."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.hf_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_model,
        torch_dtype=(torch.float16 if device == "cuda" else None),
        device_map=("auto" if device == "cuda" else None),
    )

    chat = (
        "System: You are a helpful and friendly assistant. Keep replies concise (1-2 sentences).\n"
        f"User: {prompt}\n"
        "Assistant:"
    )
    inputs = tok(chat, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    text = tok.decode(out_ids[0], skip_special_tokens=True)
    # Best-effort extraction
    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[-1]
    return text.strip()


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="All-in-one runner for Colab: STT + LLM + TTS.")

    # Input
    p.add_argument(
        "--text",
        type=str,
        default="Hello! Introduce yourself in 1-2 sentences.",
        help="Text prompt (skips STT).",
    )
    p.add_argument("--audio", type=str, default=None, help="Audio file for Whisper STT (wav/mp3/etc).")
    p.add_argument("--whisper-model", type=str, default="base", help="Whisper model name (e.g., tiny/base/small).")

    # LLM
    p.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL, e.g. http://localhost:11434")
    p.add_argument("--model", type=str, default="llama3.2:latest", help="Ollama model name.")
    p.add_argument("--no-ollama", action="store_true", help="Disable Ollama and use HF Transformers instead.")
    p.add_argument(
        "--hf-model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HF model for Transformers fallback (used automatically if Ollama isn't reachable).",
    )
    p.add_argument("--hf-max-new-tokens", type=int, default=96, help="HF generation max_new_tokens.")

    # TTS
    p.add_argument("--voice", type=str, default=None, help="Voice prompt wav for cloning (female voice etc).")
    p.add_argument("--exaggeration", type=float, default=0.5, help="Emotion intensity (0.0-1.0).")
    p.add_argument("--cfg-weight", type=float, default=0.5, help="CFG guidance strength (0.0-1.0).")
    p.add_argument("--tts-steps", type=int, default=0, help="TTS steps (speech tokens). Use 0 for auto.")
    p.add_argument("--tts-temperature", type=float, default=0.8, help="TTS sampling temperature.")
    p.add_argument("--no-tts", action="store_true", help="Skip TTS (print text only).")
    p.add_argument("--out", type=str, default="out.wav", help="Output wav path.")

    # Device
    p.add_argument("--device", type=str, default="auto", help="cuda/mps/cpu/auto for whisper+tts.")

    args = p.parse_args()

    device = _pick_device(args.device)

    # Input text
    if args.audio:
        if not os.path.exists(args.audio):
            raise FileNotFoundError(f"--audio not found: {args.audio}")
        user_text = transcribe_audio_file(args.audio, whisper_model=args.whisper_model, device=device)
    else:
        user_text = (args.text or "").strip()
        if not user_text:
            user_text = "Hello! Introduce yourself in 1-2 sentences."

    print(f"You: {user_text}")

    llm_cfg = LLMConfig(
        use_ollama=(not args.no_ollama),
        ollama_url=args.ollama_url,
        ollama_model=args.model,
        hf_model=args.hf_model,
        max_new_tokens=args.hf_max_new_tokens,
    )

    try:
        response = generate_llm_response(user_text, cfg=llm_cfg)
    except Exception as e:
        # If Ollama fails and user provided hf model, fall back automatically.
        if llm_cfg.use_ollama and args.hf_model:
            print(f"[warn] Ollama failed ({e}). Falling back to HF model: {args.hf_model}")
            llm_cfg.use_ollama = False
            response = generate_llm_response(user_text, cfg=llm_cfg)
        else:
            raise

    print(f"Assistant: {response}")

    if args.no_tts:
        return

    if args.voice and not os.path.exists(args.voice):
        print(f"[warn] --voice not found: {args.voice}. Using built-in voice.")
        args.voice = None

    tts = TextToSpeechService(device=device)
    steps = _auto_tts_steps(response, args.tts_steps)
    sr, audio = tts.synthesize(
        response,
        audio_prompt_path=args.voice,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.tts_temperature,
        max_new_tokens=steps,
    )

    # Write wav
    import soundfile as sf

    sf.write(args.out, audio, sr)
    print(f"Wrote: {args.out} (sr={sr}, steps={steps})")


if __name__ == "__main__":
    main()


