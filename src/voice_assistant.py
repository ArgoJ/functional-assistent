import sys
# specific fix for some openwakeword environments
sys.modules['coverage'] = None 

import threading
import time
import queue
import collections
import argparse
import logging

import numpy as np
import pyaudio
import whisper
import torch
import openwakeword
from openwakeword.model import Model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# --- ALSA Error Suppression ---
import ctypes
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass

class VoiceAssistant:
    """
    A standalone voice assistant script integrating a lightweight Wake Word Engine (openWakeWord)
    with a heavy Speech-to-Text model (Whisper).
    """

    def __init__(self, args):
        self.args = args
        self.running = True

        # --- Parameters from Args ---
        self.model_size = args.model_size
        self.ww_model_name = args.wake_word_model
        self.device_index = args.device_index
        self.vad_threshold = args.vad_threshold
        self.silence_limit = args.silence_limit
        self.pre_roll_seconds = args.pre_roll_seconds
        self.language = args.language
        self.debug_mode = args.debug_mode

        # --- AI Models Initialization ---
        self._init_models(self.model_size, self.ww_model_name)

        # --- Audio State ---
        self.audio_queue = queue.Queue()
        
        # Circular buffer for "pre-roll" audio
        # Chunk size 1280 @ 16kHz is 80ms. Size = seconds / 0.08
        maxlen = int(self.pre_roll_seconds / 0.08)
        self.pre_roll_buffer = collections.deque(maxlen=maxlen)

        # --- Threads ---
        self.record_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._inference_loop, daemon=True)

    def start(self):
        """Starts the recording and processing threads."""
        self.record_thread.start()
        self.process_thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stops execution gracefully."""
        logging.info("Stopping...")
        self.running = False
        # Threads are daemon, so they will die when main exits, 
        # but setting running=False helps break internal loops cleanly.

    def _init_models(self, whisper_size, ww_name):
        """Helper to load both Neural Networks safely."""
        # 1. Load Wake Word Model
        logging.info(f"Loading Wake Word Model: {ww_name}...")
        try:
            openwakeword.utils.download_models() 
            self.ww_model = Model(wakeword_models=[ww_name], inference_framework="onnx")
            self.target_ww_key = ww_name
        except Exception as e:
            logging.error(f"Failed to init openWakeWord: {e}")
            raise e

        # 2. Load Whisper
        logging.info(f"Loading Whisper Model: {whisper_size}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.asr_model = whisper.load_model(whisper_size, device=device)
            logging.info(f"Models loaded on {device}. Listening for Wake Word...")
        except Exception as e:
            logging.error(f"Failed to load Whisper: {e}")
            raise e

    def _audio_loop(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1280 

        p = pyaudio.PyAudio()

        try:
            # Handle default device index (-1)
            dev_idx = self.device_index if self.device_index >= 0 else None
            
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                            input=True, frames_per_buffer=CHUNK,
                            input_device_index=dev_idx)
        except Exception as e:
            logging.error(f"Mic Error: {e}")
            return

        is_recording_command = False
        command_frames = []
        silence_start = None

        logging.info(f"Mic listening. Threshold: {self.vad_threshold}")

        while self.running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                
                # Calculate energy (volume)
                energy = np.sqrt(np.mean(audio_int16.astype(np.float32)**2))

                # DEBUG OUTPUT
                if self.debug_mode and not is_recording_command:
                    status = "ACTIVE" if energy > self.vad_threshold else "Silence"
                    print(f"\rEnergy: {int(energy)} ({status})   ", end='', flush=True)

                if not is_recording_command:
                    # --- WATCHING ---
                    self.pre_roll_buffer.append(data)
                    prediction = self.ww_model.predict(audio_int16)
                    score = prediction.get(self.target_ww_key, 0.0)
                    
                    if score > 0.5: 
                        logging.info(f"\n--- WAKE WORD DETECTED ({score:.2f}) ---")
                        is_recording_command = True
                        command_frames.extend(self.pre_roll_buffer)
                        self.pre_roll_buffer.clear()
                        silence_start = None
                else:
                    # --- RECORDING ---
                    command_frames.append(data)
                    
                    if energy < self.vad_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        
                        duration_silent = time.time() - silence_start
                        if self.debug_mode:
                            print(f"\rRecording... Silence: {duration_silent:.1f}s / {self.silence_limit}s", end='', flush=True)

                        if duration_silent > self.silence_limit:
                            logging.info("\nCommand finished. Processing...")
                            full_audio = b''.join(command_frames)
                            self.audio_queue.put(full_audio)
                            command_frames = []
                            is_recording_command = False
                            self.ww_model.reset()
                    else:
                        # User is speaking -> Reset silence timer
                        if silence_start is not None and self.debug_mode:
                            print(f"\rRecording... Voice detected! Timer reset.", end='', flush=True)
                        silence_start = None

            except Exception as e:
                logging.error(f"Audio loop error: {e}")
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _inference_loop(self):
        while self.running:
            try:
                audio_bytes = self.audio_queue.get(timeout=1.0)
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                result = self.asr_model.transcribe(
                    audio_np, 
                    fp16=torch.cuda.is_available(), 
                    language=self.language 
                )
                
                # Cleanup typical hallucinations from Whisper
                text = result['text'].strip().lstrip('Hey Jarvis,. ').strip(',:;.!?')

                if text:
                    logging.info(f"TRANSCRIPTION: '{text}'")
                    # Here you can add logic to send this text to another API or System
                else:
                    logging.warning("Transcribed empty text.")
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Inference error: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Voice Assistant with Wake Word and Whisper")
    
    parser.add_argument('--model_size', type=str, default='base', 
                        help='Whisper model size (tiny, base, small, medium, large)')
    parser.add_argument('--wake_word_model', type=str, default='hey_jarvis_v0.1', 
                        help='OpenWakeWord model name')
    parser.add_argument('--device_index', type=int, default=-1, 
                        help='Microphone device index (default: system default)')
    parser.add_argument('--vad_threshold', type=int, default=60, 
                        help='Energy threshold for voice activity detection')
    parser.add_argument('--silence_limit', type=float, default=1.2, 
                        help='Seconds of silence before stopping recording')
    parser.add_argument('--pre_roll_seconds', type=float, default=1.0, 
                        help='Seconds of audio to keep in buffer before trigger')
    parser.add_argument('--language', type=str, default='en', 
                        help='Language code for Whisper')
    parser.add_argument('--debug_mode', action='store_true', 
                        help='Show energy levels and recording status in console')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    assistant = VoiceAssistant(args)
    assistant.start()