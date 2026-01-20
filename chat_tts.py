import ollama
from xtts import tts_with_live_playback, speaker_wav

OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.2:latest"

SYSTEM_PROMPT = """You are a helpful assistant. Format your responses with these rules:
1. Break your response into small chunks of 13-15 words maximum
2. Add "#" after each meaningful chunk
3. Use simple, everyday words
4. Make each chunk emotionally complete and satisfying on its own
5. Keep chunks natural and conversational
6. Keep overall response short and funny.
Example format:
Hello there friend# How are you doing today? # I hope you are feeling well # What can I help you with? # """

client = ollama.Client(host=OLLAMA_HOST)
conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

print("Chat with Ollama + TTS")
print("Type your message and press Enter. Type 'quit' or 'exit' to stop.\n")

while True:
    try:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        conversation_history.append({"role": "user", "content": user_input})
        
        print("Thinking...")
        response = client.chat(
            model=MODEL,
            messages=conversation_history
        )
        
        assistant_message = response['message']['content']
        conversation_history.append({"role": "assistant", "content": assistant_message})
        
        assistant_message = response['message']['content']
        print(f"\nAssistant: {assistant_message}\n")
        
        if assistant_message.strip():
            tts_with_live_playback(
                text=assistant_message,
                speaker_wav=speaker_wav,
                language="en",
                speed = 1
            )
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue

