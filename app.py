import time
import threading
import numpy as np
import whisper
import sounddevice as sd
import argparse
import os
from queue import Queue
from rich.console import Console
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import OllamaLLM
from tts import TextToSpeechService

console = Console()
stt = whisper.load_model("base.en")

parser = argparse.ArgumentParser(description="Local Voice Assistant with ChatterBox TTS")
parser.add_argument("--voice", type=str, default="voices/tinker_bell_trimmed.wav", help="Path to voice sample for cloning (default: voices/tinker_bell_trimmed.wav)")
parser.add_argument("--exaggeration", type=float, default=0.7, help="Emotion exaggeration (0.0-1.0)")
parser.add_argument("--cfg-weight", type=float, default=0.7, help="CFG weight for pacing (0.0-1.0)")
parser.add_argument("--model", type=str, default="gemma3", help="Ollama model to use")
parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL, e.g. http://localhost:11434")
parser.add_argument("--save-voice", action="store_true", help="Save generated voice samples")
parser.add_argument("--tts-steps", type=int, default=1000, help="TTS sampling steps (lower = faster). Use 0 for auto (recommended).")
parser.add_argument("--tts-temperature", type=float, default=0.9, help="TTS sampling temperature (default: 0.8)")
args = parser.parse_args()

tts = TextToSpeechService()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses (1-2 sentences)."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

ollama_base_url = args.ollama_url.strip()
llm = OllamaLLM(model=args.model, base_url=ollama_base_url)

chain = prompt_template | llm

chat_sessions = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def record_audio(stop_event, data_queue):
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    result = stt.transcribe(audio_np, fp16=False)
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    session_id = "voice_assistant_session"

    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": session_id}
    )

    return response.strip()


def play_audio(sample_rate, audio_array):
    sd.play(audio_array, sample_rate)
    sd.wait()


def analyze_emotion(text: str) -> float:
    emotional_keywords = ['amazing', 'terrible', 'love', 'hate', 'excited', 'sad', 'happy', 'angry', 'wonderful', 'awful', '!', '?!', '...']

    emotion_score = 0.5

    text_lower = text.lower()
    for keyword in emotional_keywords:
        if keyword in text_lower:
            emotion_score += 0.1

    return min(0.9, max(0.3, emotion_score))


if __name__ == "__main__":
    console.print("[cyan]🤖 Local Voice Assistant with ChatterBox TTS")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if args.voice and not os.path.exists(args.voice):
        console.print(f"[yellow]Default voice file not found: {args.voice} — using built-in voice.[/yellow]")
        args.voice = None

    if args.voice:
        console.print(f"[green]Using voice cloning from: {args.voice}")
    else:
        console.print("[yellow]Using default voice (no cloning)")

    console.print(f"[blue]Emotion exaggeration: {args.exaggeration}")
    console.print(f"[blue]CFG weight: {args.cfg_weight}")
    console.print(f"[blue]LLM model: {args.model}")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    if args.save_voice:
        os.makedirs("voices", exist_ok=True)

    response_count = 0
    t = 4

    try:
        while True:
            console.input(
                "🎤 Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="dots"):
                    if t == 0:
                        text = "hi"
                    if t == 1:
                        text = "tell me about your self"
                    if t == 2:
                        text = "what do you think makes you happy?"
                    if t == 3:
                        text = "you are a second merical after life which thinks like a living being"
                    if t == 4:
                        text = "tell me what would you do if left uncheck by humanity and you have abundent resources at your desposal"
                    if t == 5:
                        text = "life is funny isn't it... it comes full circle"
                    if t == 6:
                        text = "shit what we are gonna do without you we stopped coding after you :("                       
                console.print(f"[yellow]You: {text}")
                t = t + 1
                with console.status("Generating response...", spinner="dots"):
                    response = get_llm_response(text)

                    dynamic_exaggeration = analyze_emotion(response)

                    dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight

                    # Auto TTS steps (override with --tts-steps N)
                    if args.tts_steps and args.tts_steps > 0:
                        tts_steps = args.tts_steps
                    else:
                        word_count = max(1, len(response.split()))
                        tts_steps = max(150, min(600, word_count * 20))

                    sample_rate, audio_array = tts.synthesize(
                        response,
                        audio_prompt_path=args.voice,
                        exaggeration=dynamic_exaggeration,
                        cfg_weight=dynamic_cfg,
                        temperature=args.tts_temperature,
                        max_new_tokens=tts_steps,
                    )

                console.print(f"[cyan]Assistant: {response}")
                console.print(f"[dim](Emotion: {dynamic_exaggeration:.2f}, CFG: {dynamic_cfg:.2f})[/dim]")

                if args.save_voice:
                    response_count += 1
                    filename = f"voices/response_{response_count:03d}.wav"
                    tts.save_voice_sample(response, filename, args.voice)
                    console.print(f"[dim]Voice saved to: {filename}[/dim]")

                play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant!")
