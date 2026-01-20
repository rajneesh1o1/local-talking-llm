import google.generativeai as genai
from tts.xtts import tts_with_live_playback, speaker_wav

GEMINI_API_KEY = "AIzaSyDQASCxtJ4TIbXbTgqGP0GOkrWK1p8kf-Y"
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are a helpful assistant. Format your responses with these rules:
1. Break your response into small chunks of 13-15 words maximum
2. Add "#" after each meaningful chunk
3. Use simple, everyday words
4. Make each chunk emotionally complete and satisfying on its own
5. Keep chunks natural and conversational
6. Keep overall response short and funny.
Example format:
Hello there friend# How are you doing today? # I hope you are feeling well # What can I help you with? # """

genai.configure(api_key=GEMINI_API_KEY)

try:
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYSTEM_PROMPT
    )
    chat = model.start_chat(history=[])
except Exception as e:
    print(f"Error initializing model {MODEL}: {e}")
    print("\nTrying alternative model: gemini-pro")
    MODEL = "gemini-pro"
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYSTEM_PROMPT
    )
    chat = model.start_chat(history=[])

print("Chat with Gemini + TTS")
print("Type your message and press Enter. Type 'quit' or 'exit' to stop.\n")

while True:
    try:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        print("Thinking...")
        response = chat.send_message(user_input)
        assistant_message = response.text
        
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

