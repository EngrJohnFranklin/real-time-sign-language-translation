"""
Speech Handler Module for Real-Time Sign Language Translation.

This module provides speech-to-text (using Vosk) and text-to-speech (using pyttsx3)
capabilities for the sign language translation system. All processing is done offline
with no internet dependency.
"""

import json
import os
import logging
import threading
import time
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import queue

import vosk
import pyttsx3
import pyaudio

logger = logging.getLogger(__name__)


class SpeechLanguage(Enum):
    """Supported languages for speech recognition."""
    ENGLISH = "en"
    FILIPINO = "fil"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    DEFAULT = "en"


@dataclass
class SpeechResult:
    """Data class to hold speech recognition results."""
    text: str
    confidence: float
    is_final: bool
    language: str = "en"
    
    def __str__(self) -> str:
        """String representation of speech result."""
        status = "Final" if self.is_final else "Partial"
        return f"{status}: '{self.text}' (confidence: {self.confidence:.2f})"


class VoskSpeechRecognizer:
    """
    Offline speech-to-text recognizer using Vosk library.
    
    Vosk is a lightweight offline speech recognition engine that works without
    internet connectivity. It uses acoustic models for real-time recognition.
    """
    
    # Audio configuration
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 4096
    CHANNELS = 1
    MIN_PARTIAL_CHARS = 2
    MIN_FINAL_CHARS = 2
    
    def __init__(self, model_path: Optional[str] = None, language: SpeechLanguage = SpeechLanguage.ENGLISH):
        """
        Initialize Vosk speech recognizer.
        
        Args:
            model_path: Path to Vosk model directory. If None, uses default model.
            language: Language for recognition (SpeechLanguage enum).
            
        Raises:
            RuntimeError: If model cannot be loaded or no microphone available.
        """
        self.model_path = model_path
        self.language = language
        self.model = None
        self.recognizer = None
        self.audio_stream = None
        self.pyaudio_instance = None
        self.is_listening = False
        self.recognition_thread = None
        self.audio_queue = queue.Queue()
        self.result_callback = None
        self._last_partial_text = ""
        self._last_final_text = ""
        
        try:
            self._initialize_model()
            self._initialize_audio()
            logger.info(f"VoskSpeechRecognizer initialized with language: {language.value}")
        except Exception as e:
            logger.error(f"Error initializing VoskSpeechRecognizer: {e}")
            self.cleanup()
            raise RuntimeError(f"Failed to initialize speech recognizer: {e}")
    
    def _initialize_model(self):
        """
        Load Vosk model from disk or use default model.
        
        The model contains acoustic and language data needed for recognition.
        Models must be downloaded separately from Vosk website.
        
        Raises:
            RuntimeError: If model cannot be loaded.
        """
        try:
            if self.model_path:
                # Try to use provided model path
                self.model = vosk.Model(self.model_path)
            else:
                # Select model candidates by language with fallback to default English models.
                model_candidates = self._get_model_candidates_for_language(self.language)
                model_found = False

                for candidate in model_candidates:
                    if os.path.exists(candidate):
                        logger.info(f"Loading Vosk model from: {candidate}")
                        self.model = vosk.Model(candidate)
                        model_found = True
                        break

                if not model_found:
                    raise RuntimeError(
                        f"No Vosk model found for language '{self.language.value}'. "
                        f"Tried: {', '.join(model_candidates)}"
                    )
            
            self.recognizer = vosk.KaldiRecognizer(self.model, self.SAMPLE_RATE)
            logger.debug("Vosk model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Vosk model: {e}")
            raise RuntimeError(f"Could not load Vosk model. {str(e)}")

    @staticmethod
    def _get_model_candidates_for_language(language: SpeechLanguage) -> List[str]:
        """
        Get language-specific Vosk model candidate paths.

        Args:
            language: Selected speech language.

        Returns:
            Ordered candidate model directories.
        """
        english_candidates = [
            "model-en-us",
            "model-en",
            "model",
            "models/vosk/model-en-us",
            "models/vosk/model"
        ]

        filipino_candidates = [
            "model-fil",
            "model-tl",
            "model-ph",
            "models/vosk/model-fil",
            "models/vosk/model-tl"
        ]

        if language == SpeechLanguage.FILIPINO:
            return filipino_candidates + english_candidates

        return english_candidates
    
    def _initialize_audio(self):
        """
        Initialize PyAudio for microphone input.
        
        Sets up audio stream with proper sample rate, channels, and chunk size
        for real-time speech recognition.
        
        Raises:
            RuntimeError: If no microphone available.
        """
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Verify microphone is available
            device_count = self.pyaudio_instance.get_device_count()
            if device_count == 0:
                raise RuntimeError("No audio input devices available")
            
            # Open audio stream for recording
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            logger.debug(f"Audio stream initialized: {device_count} devices available")
        except Exception as e:
            logger.error(f"Error initializing audio: {e}")
            raise RuntimeError(f"Could not initialize audio input: {e}")
    
    def start_listening(self, callback: Optional[Callable[[SpeechResult], None]] = None):
        """
        Start listening for speech input in background thread.
        
        Speech recognition runs continuously in a separate thread and calls
        the callback function for each recognition result.
        
        Args:
            callback: Function to call with SpeechResult when recognition completes.
                     Callback receives: SpeechResult object.
        """
        if self.is_listening:
            logger.warning("Already listening for speech")
            return
        
        try:
            self.result_callback = callback
            self.is_listening = True
            self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
            self.recognition_thread.start()
            logger.info("Speech recognition started")
        except Exception as e:
            logger.error(f"Error starting speech recognition: {e}")
            self.is_listening = False
            raise
    
    def stop_listening(self):
        """
        Stop listening for speech input.
        
        Stops the recognition thread and cleans up audio resources.
        """
        try:
            self.is_listening = False
            
            if self.recognition_thread and self.recognition_thread.is_alive():
                self.recognition_thread.join(timeout=2.0)
            
            logger.info("Speech recognition stopped")
        except Exception as e:
            logger.error(f"Error stopping speech recognition: {e}")
    
    def _recognition_loop(self):
        """
        Main recognition loop running in background thread.
        
        Continuously reads audio chunks and processes them with Vosk recognizer.
        Calls result callback when partial or final results are available.
        This runs in a separate thread and should not be called directly.
        """
        try:
            if not self.audio_stream or not self.recognizer:
                logger.error("Audio stream or recognizer not initialized")
                return
            
            while self.is_listening:
                try:
                    # Read audio chunk from microphone
                    data = self.audio_stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    
                    if not data:
                        continue
                    
                    # Process audio chunk with Vosk
                    if self.recognizer.AcceptWaveform(data):
                        # Final result
                        result_json = self.recognizer.Result()
                        result = self._parse_recognition_result(result_json, is_final=True)
                        
                        if result and self.result_callback:
                            if result.text.lower() == self._last_final_text.lower():
                                continue
                            self._last_final_text = result.text
                            self._last_partial_text = ""
                            self.result_callback(result)
                    else:
                        # Partial result
                        partial_json = self.recognizer.PartialResult()
                        result = self._parse_recognition_result(partial_json, is_final=False)
                        
                        if result and self.result_callback:
                            if result.text.lower() == self._last_partial_text.lower():
                                continue
                            self._last_partial_text = result.text
                            self.result_callback(result)
                
                except Exception as e:
                    logger.error(f"Error in recognition loop: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Critical error in recognition loop: {e}")
            self.is_listening = False
    
    def _parse_recognition_result(self, result_json: str, is_final: bool) -> Optional[SpeechResult]:
        """
        Parse JSON result from Vosk recognizer.
        
        Vosk returns results in JSON format containing text and confidence scores.
        
        Args:
            result_json: JSON string from Vosk recognizer
            is_final: Whether this is a final or partial result
            
        Returns:
            SpeechResult object with recognized text, or None if parsing fails
        """
        try:
            if not result_json:
                return None
            
            result_dict = json.loads(result_json)
            
            # Final result format typically contains text + optional word-level confidence list.
            if is_final and 'text' in result_dict:
                text = result_dict.get('text', '')
                word_items = result_dict.get('result', []) or []
                confidence_values = [item.get('conf', 1.0) for item in word_items if isinstance(item.get('conf', 1.0), (int, float))]
                confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 1.0
            
            # Handle partial result format
            elif 'partial' in result_dict:
                text = result_dict['partial']
                confidence = 0.5  # Partial results have lower confidence
            
            # Handle fallback result text key
            elif 'text' in result_dict:
                text = result_dict.get('text', '')
                confidence = 1.0
            
            else:
                return None
            
            text = text.strip()
            if not text:
                return None

            # Ignore silence/noise-like ultra-short outputs.
            min_chars = self.MIN_FINAL_CHARS if is_final else self.MIN_PARTIAL_CHARS
            if len(text) < min_chars:
                return None
            
            return SpeechResult(
                text=text,
                confidence=confidence,
                is_final=is_final,
                language=self.language.value
            )
        
        except json.JSONDecodeError as e:
            logger.debug(f"Could not parse recognition result: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing recognition result: {e}")
            return None
    
    def get_recognition_text(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Wait for and retrieve final recognition result (blocking call).
        
        Args:
            timeout: Maximum time to wait in seconds. None = wait indefinitely.
            
        Returns:
            Recognized text string, or None if timeout or error occurs.
        """
        try:
            final_text_event = threading.Event()
            final_text = None
            
            # Capture final results in callback
            def capture_final(result: SpeechResult):
                nonlocal final_text
                if result.is_final:
                    final_text = result.text
                    final_text_event.set()
            
            old_callback = self.result_callback
            self.result_callback = capture_final
            
            # Wait for final result using threading.Event (efficient synchronization)
            # This is more efficient than polling with time.sleep()
            final_text_event.wait(timeout=timeout)
            
            if final_text_event.is_set():
                logger.debug(f"Recognition result: '{final_text}'")
            else:
                logger.warning(f"Speech recognition timeout after {timeout}s")
            
            self.result_callback = old_callback
            return final_text
        
        except Exception as e:
            logger.error(f"Error getting recognition text: {e}")
            return None
    
    def cleanup(self):
        """
        Clean up audio resources and stop listening.
        
        Should be called before application exit to properly release
        microphone and audio resources.
        """
        try:
            self.stop_listening()
            
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            
            logger.info("VoskSpeechRecognizer cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class TextToSpeechEngine:
    """
    Text-to-speech engine using pyttsx3.
    
    pyttsx3 is a cross-platform offline text-to-speech library that works
    without internet connectivity or cloud dependencies.
    """
    
    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Initialize text-to-speech engine.
        
        Args:
            rate: Speech rate in words per minute (50-300). Default: 150.
            volume: Volume level (0.0-1.0). Default: 1.0.
            
        Raises:
            RuntimeError: If TTS engine cannot be initialized.
        """
        try:
            self.engine = pyttsx3.init()
            self.rate = rate
            self.volume = volume
            self.is_speaking = False
            
            # Set speech rate
            self.engine.setProperty('rate', rate)
            
            # Set volume
            self.engine.setProperty('volume', volume)
            
            # Get available voices
            self.voices = self.engine.getProperty('voices')
            if not self.voices:
                logger.warning("No voices available for TTS")
            
            logger.info(f"TextToSpeechEngine initialized: rate={rate}, volume={volume}")
        except Exception as e:
            logger.error(f"Error initializing TTS engine: {e}")
            raise RuntimeError(f"Failed to initialize text-to-speech: {e}")
    
    def set_voice(self, voice_id: int = 0):
        """
        Set the voice for speech synthesis.
        
        Args:
            voice_id: Index of voice to use (0 = first available voice).
                     Use get_available_voices() to see options.
        """
        try:
            if voice_id < len(self.voices):
                self.engine.setProperty('voice', self.voices[voice_id].id)
                logger.debug(f"Voice set to: {self.voices[voice_id].name}")
            else:
                logger.warning(f"Voice ID {voice_id} not available")
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
    
    def set_rate(self, rate: int):
        """
        Set speech rate.
        
        Args:
            rate: Words per minute (50-300). Default is around 150.
        """
        try:
            if 50 <= rate <= 300:
                self.rate = rate
                self.engine.setProperty('rate', rate)
                logger.debug(f"Speech rate set to {rate} WPM")
            else:
                logger.warning(f"Speech rate {rate} out of valid range (50-300)")
        except Exception as e:
            logger.error(f"Error setting speech rate: {e}")
    
    def set_volume(self, volume: float):
        """
        Set speech volume.
        
        Args:
            volume: Volume level (0.0-1.0). 0.0 = silent, 1.0 = maximum.
        """
        try:
            if 0.0 <= volume <= 1.0:
                self.volume = volume
                self.engine.setProperty('volume', volume)
                logger.debug(f"Volume set to {volume}")
            else:
                logger.warning(f"Volume {volume} out of valid range (0.0-1.0)")
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
    
    def speak(self, text: str, wait: bool = True):
        """
        Synthesize and speak text.
        
        Args:
            text: Text to synthesize and speak.
            wait: If True, blocks until speech finishes. If False, speaks asynchronously.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to speak()")
            return
        
        try:
            self.is_speaking = True
            self.engine.say(text)
            
            if wait:
                self.engine.runAndWait()
            else:
                # Non-blocking speech (platform dependent)
                self.engine.startLoop(False)
                while self.engine.isBusy():
                    self.engine.iterate()
            
            self.is_speaking = False
            logger.debug(f"Spoken: '{text}'")
        except Exception:
            logger.exception("Error speaking text: %r", text[:40])
            self.is_speaking = False
    
    def speak_async(self, text: str, sequence_id: int = 0,
                    is_current: Optional[Callable[[], bool]] = None):
        """
        Speak text asynchronously without blocking.

        A new pyttsx3 engine is created inside the worker thread on every call.
        This is required on Windows because SAPI5 is a COM Single-Threaded
        Apartment (STA) object: the engine created in __init__ (main thread)
        cannot be used from a different thread — engine.runAndWait() returns
        immediately without producing audio.  Creating the engine in the same
        thread that calls runAndWait() correctly initialises the COM apartment
        and allows the speech event loop to pump messages.

        If is_current() is provided and returns False, this speech is skipped
        to prevent stale queued speech from playing after newer recognition.

        Args:
            text: Text to synthesize and speak.
            sequence_id: Optional sequence number for tracking.
            is_current: Optional callable that returns True if this speech is still current.
        """
        def _worker(text: str, rate: int, volume: float, seq_id: int,
                    is_curr: Optional[Callable[[], bool]]) -> None:
            try:
                # Skip if a newer speech has been requested (stale suppression)
                if is_curr and not is_curr():
                    logger.debug("TTS worker: skipping stale speech (sequence %d)", seq_id)
                    return
                
                engine = pyttsx3.init()
                engine.setProperty('rate', rate)
                engine.setProperty('volume', volume)
                logger.debug("TTS worker: engine.say(%r) [sequence %d]", text[:40], seq_id)
                engine.say(text)
                logger.debug("TTS worker: engine.runAndWait() starting [sequence %d]", seq_id)
                engine.runAndWait()
                logger.debug("TTS worker: engine.runAndWait() completed [sequence %d]", seq_id)
            except Exception:
                logger.exception("TTS worker thread error for %r", text[:40])

        try:
            thread = threading.Thread(
                target=_worker,
                args=(text, self.rate, self.volume, sequence_id, is_current),
                daemon=True,
            )
            thread.start()
            logger.debug("TTS async thread started for %r [sequence %d]", text[:40], sequence_id)
        except Exception:
            logger.exception("Error starting async speech thread")
    
    def get_available_voices(self) -> List[dict]:
        """
        Get list of available voices.
        
        Returns:
            List of dictionaries with voice information (id, name, languages, etc.)
        """
        try:
            voices_info = []
            for i, voice in enumerate(self.voices):
                voices_info.append({
                    'id': i,
                    'name': voice.name,
                    'voice_id': voice.id,
                    'languages': getattr(voice, 'languages', [])
                })
            return voices_info
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
    
    def save_to_file(self, text: str, filename: str):
        """
        Save synthesized speech to audio file.
        
        Args:
            text: Text to synthesize.
            filename: Path to save audio file (e.g., 'output.wav' or 'output.mp3').
        """
        try:
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
            logger.info(f"Audio saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
    
    def stop(self):
        """Stop current speech playback immediately."""
        try:
            self.engine.stop()
            self.is_speaking = False
            logger.debug("Speech stopped")
        except Exception as e:
            logger.error(f"Error stopping speech: {e}")
    
    def cleanup(self):
        """Clean up TTS resources."""
        try:
            self.stop()
            logger.info("TextToSpeechEngine cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class SpeechHandler:
    """
    Main speech handler combining speech recognition and text-to-speech.
    
    Provides a unified interface for bidirectional speech communication
    with full offline capability.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 language: SpeechLanguage = SpeechLanguage.ENGLISH,
                 tts_rate: int = 150,
                 tts_volume: float = 1.0):
        """
        Initialize speech handler with both recognition and TTS.
        
        Args:
            model_path: Path to Vosk model directory.
            language: Language for speech recognition.
            tts_rate: Text-to-speech rate in WPM.
            tts_volume: Text-to-speech volume (0.0-1.0).
            
        Raises:
            RuntimeError: If speech handler cannot be initialized.
        """
        self.recognizer = None
        self.tts_engine = None
        
        try:
            # Initialize speech recognizer
            self.recognizer = VoskSpeechRecognizer(model_path, language)
            
            # Initialize text-to-speech
            self.tts_engine = TextToSpeechEngine(tts_rate, tts_volume)
            
            logger.info("SpeechHandler initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SpeechHandler: {e}")
            self.cleanup()
            raise
    
    def listen_and_recognize(self, callback: Optional[Callable[[SpeechResult], None]] = None):
        """
        Start listening for speech and recognize it.
        
        Args:
            callback: Function to call with SpeechResult for each recognition.
        """
        try:
            self.recognizer.start_listening(callback)
        except Exception:
            logger.exception("Error starting recognition")
            raise
    
    def stop_listening(self):
        """Stop listening for speech input."""
        try:
            self.recognizer.stop_listening()
        except Exception:
            logger.exception("Error stopping recognition")
            raise
    
    def speak(self, text: str, wait: bool = True):
        """
        Speak text using text-to-speech.
        
        Args:
            text: Text to speak.
            wait: If True, blocks until speech finishes.
        """
        try:
            self.tts_engine.speak(text, wait)
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
    
    def speak_async(self, text: str, sequence_id: int = 0, 
                    is_current: Optional[Callable[[], bool]] = None):
        """
        Speak text asynchronously without blocking.
        
        Args:
            text: Text to speak.
            sequence_id: Optional sequence number to track recognition order.
            is_current: Optional callable that returns True if this speech is still current.
                       If returns False, execution is skipped (stale speech suppression).
        """
        try:
            self.tts_engine.speak_async(text, sequence_id=sequence_id, is_current=is_current)
        except Exception:
            logger.exception("Error in async speech")
            raise
    
    def cleanup(self):
        """Clean up all speech resources."""
        try:
            if self.recognizer:
                self.recognizer.cleanup()
            
            if self.tts_engine:
                self.tts_engine.cleanup()
            
            logger.info("SpeechHandler cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example usage and testing
if __name__ == "__main__":
    import time
    
    try:
        # Initialize speech handler
        handler = SpeechHandler()
        
        print("Speech Handler Example")
        print("=" * 50)
        
        # Test text-to-speech
        print("\nTesting Text-to-Speech...")
        handler.speak("Hello, this is a test of the text to speech system.", wait=True)
        
        # Test speech recognition
        print("\nStarting speech recognition...")
        print("Speak something for 5 seconds...")
        
        recognition_results = []
        
        def on_speech_result(result: SpeechResult):
            recognition_results.append(result)
            print(f"Recognition: {result}")
        
        handler.listen_and_recognize(callback=on_speech_result)
        
        # Listen for 5 seconds
        time.sleep(5)
        handler.stop_listening()
        
        # Speak the recognized text back
        if recognition_results:
            last_result = recognition_results[-1]
            if last_result.text:
                print(f"\nRecognized: {last_result.text}")
                print("Speaking back the recognized text...")
                handler.speak(f"I heard you say: {last_result.text}")
        
        # Clean up
        handler.cleanup()
        print("\nExample completed successfully")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        print(f"Error: {e}")
