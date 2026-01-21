"""Voice-to-text functionality using microphone input."""

import os
import logging
import speech_recognition as sr

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

# Initialize recognizer
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Adjust for ambient noise (one-time setup)
_ambient_noise_adjusted = False


def adjust_for_ambient_noise():
    """Adjust recognizer for ambient noise (call once at startup)."""
    global _ambient_noise_adjusted
    if not _ambient_noise_adjusted:
        try:
            logger.info("🎤 Adjusting microphone for ambient noise...")
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
            _ambient_noise_adjusted = True
            logger.info("✅ Microphone adjusted for ambient noise")
        except Exception as e:
            logger.warning(f"⚠️ Could not adjust for ambient noise: {e}")


def voice_to_text(timeout=5, phrase_time_limit=10) -> str:
    """
    Convert voice input from microphone to text.
    
    Args:
        timeout: Maximum seconds to wait for speech to start
        phrase_time_limit: Maximum seconds to listen for speech
    
    Returns:
        Transcribed text string, or empty string if no speech detected or error
    """
    try:
        adjust_for_ambient_noise()
        
        logger.info("🎤 Listening... (speak now)")
        with microphone as source:
            # Listen for audio with timeout
            audio = recognizer.listen(
                source, 
                timeout=timeout, 
                phrase_time_limit=phrase_time_limit
            )
        
        logger.info("🔄 Processing speech...")
        
        # Try Google Speech Recognition (free, requires internet)
        try:
            text = recognizer.recognize_google(audio)
            logger.info(f"✅ Recognized: {text}")
            return text
        except sr.UnknownValueError:
            logger.warning("⚠️ Could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"❌ Error with speech recognition service: {e}")
            # Fallback to offline recognition if available
            try:
                logger.info("🔄 Trying offline recognition...")
                text = recognizer.recognize_sphinx(audio)
                logger.info(f"✅ Recognized (offline): {text}")
                return text
            except:
                logger.error("❌ Offline recognition also failed")
                return ""
                
    except sr.WaitTimeoutError:
        logger.warning("⚠️ No speech detected within timeout period")
        return ""
    except Exception as e:
        logger.error(f"❌ Error in voice-to-text: {e}", exc_info=True)
        return ""


def voice_to_text_with_fallback(timeout=5, phrase_time_limit=10) -> str:
    """
    Convert voice to text with multiple fallback options.
    Tries: Google (online) -> Sphinx (offline) -> Whisper (if available)
    """
    try:
        adjust_for_ambient_noise()
        
        logger.info("🎤 Listening... (speak now)")
        with microphone as source:
            audio = recognizer.listen(
                source, 
                timeout=timeout, 
                phrase_time_limit=phrase_time_limit
            )
        
        logger.info("🔄 Processing speech...")
        
        # Try Google Speech Recognition first (best quality, requires internet)
        try:
            text = recognizer.recognize_google(audio)
            logger.info(f"✅ Recognized (Google): {text}")
            return text
        except sr.UnknownValueError:
            logger.debug("Google couldn't understand audio, trying fallback...")
        except sr.RequestError as e:
            logger.debug(f"Google service error: {e}, trying fallback...")
        
        # Try Whisper (offline, high quality, slower)
        try:
            import whisper
            import tempfile
            logger.info("🔄 Trying Whisper (offline)...")
            # Save audio to temporary file for Whisper
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_path = tmp_file.name
                with open(tmp_path, "wb") as f:
                    f.write(audio.get_wav_data())
            
            try:
                model = whisper.load_model("base")  # Use base model for speed
                result = model.transcribe(tmp_path, language="en")
                text = result["text"].strip()
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                # Filter out common false positives
                if text and text.lower() not in ["true", "false", "yes", "no", "okay", "ok"]:
                    logger.info(f"✅ Recognized (Whisper): {text}")
                    return text
                elif text:
                    logger.debug(f"Whisper returned likely false positive: '{text}', trying Sphinx...")
            except Exception as e:
                # Clean up temp file on error
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                raise e
        except ImportError:
            logger.debug("Whisper not available, trying Sphinx...")
        except Exception as e:
            logger.debug(f"Whisper error: {e}, trying Sphinx...")
        
        # Try Sphinx (offline, lower quality, faster)
        try:
            logger.info("🔄 Trying Sphinx (offline)...")
            text = recognizer.recognize_sphinx(audio)
            if text:
                logger.info(f"✅ Recognized (Sphinx): {text}")
                return text
        except Exception as e:
            logger.debug(f"Sphinx error: {e}")
        
        logger.warning("⚠️ Could not recognize speech with any method")
        return ""
        
    except sr.WaitTimeoutError:
        logger.warning("⚠️ No speech detected within timeout period")
        return ""
    except Exception as e:
        logger.error(f"❌ Error in voice-to-text: {e}", exc_info=True)
        return ""

