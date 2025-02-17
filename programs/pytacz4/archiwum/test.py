import subprocess
import tempfile
from gtts import gTTS
import os
import time
from playsound import playsound


# Test variables
USE_GOOGLE_TTS = True  # Set to True to use Google TTS
system_name = "Windows"  # Change this to "Linux" or "macOS" as needed
voice_language = "en"  # Language for TTS
text = "Hello, this is a test for Google Text-to-Speech."

def speak(text, voice_language):
    # Set main working directories
    main_folder = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory if running as a script
    tempfile.tempdir = os.path.join(main_folder, "temp")
    global USE_GOOGLE_TTS
    
    # Handling split by "|"
    text = text.split("|")[0]

    if USE_GOOGLE_TTS:
        try:
            tts = gTTS(text, lang=voice_language)
            # Create a temporary audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_filename = temp_audio.name
                tts.save(temp_filename)
                print(f"Saving speech to: {temp_filename}")
            
            # Play the audio file
            playsound(temp_filename)
            
            # Add a small delay before deleting the file
            time.sleep(1)  # Optional: additional delay before file removal
            os.remove(temp_filename)
            print(f"File {temp_filename} deleted successfully.")
        except Exception as e:
            print(f"Błąd podczas syntezowania mowy (Google TTS): {e}")
    else:
        try:
            os.system(f"espeak-ng -v {voice_language} \"{text}\"")
        except Exception as e:
            print(f"Błąd podczas syntezowania mowy (espeak-ng): {e}")


# Run the function for testing
speak(text, voice_language)
