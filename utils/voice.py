import pyttsx3
import speech_recognition as sr
import streamlit as st


def get_voice_input():
    """
    Simple STT using Google Speech Recognition.
    Uses system microphone directly.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(
                source, duration=0.5)
            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=15
            )
        text = recognizer.recognize_google(audio)
        return text, None
    except sr.WaitTimeoutError:
        return None, "No speech detected. Try again."
    except sr.UnknownValueError:
        return None, "Could not understand. Speak clearly."
    except sr.RequestError:
        return None, "Speech service unavailable."
    except OSError:
        return None, "Microphone not found."
    except Exception as e:
        return None, str(e)


def speak_text(text):
    """
    Simple offline TTS using pyttsx3.
    Works completely offline on Windows.
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        clean_text = text[:300]
        engine.say(clean_text)
        engine.runAndWait()
        engine.stop()
        del engine
    except Exception:
        pass
