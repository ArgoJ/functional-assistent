import sys
import os
import queue
import threading
import time
import collections
import logging
import argparse
import tempfile
import wave
import subprocess
import json
import re

import inspect
import numpy as np
import pyaudio
import openwakeword
from openwakeword.model import Model

# Optimized: No transformers dependency needed
from llama_cpp import Llama

# Import your tools definitions
try:
    from functional_gemma import tools
except ImportError:
    # If running from src directly
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'functional_gemma'))
    import tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# --- ALSA Error Suppression (same as original) ---
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

def py_function_to_json(func):
    """
    Simple helper to generate JSON schema from python function without transformers.
    Assumes typing hints and docstrings are present.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null"
    }

    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    description = doc.split("\n")[0]  # Simple first line description

    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for name, param in sig.parameters.items():
        param_type = type_map.get(param.annotation, "string")
        parameters["properties"][name] = {
            "type": param_type
        }
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)
    
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": parameters
        }
    }

class RaspiAssistant:
    def __init__(self, args):
        self.args = args
        self.running = True

        # Configuration
        self.ww_model_name = args.wake_word_model
        self.device_index = args.device_index
        self.vad_threshold = args.vad_threshold
        self.silence_limit = args.silence_limit
        self.pre_roll_seconds = args.pre_roll_seconds
        self.debug_mode = args.debug_mode
        self.whisper_bin = args.whisper_bin_path
        self.whisper_model_path = args.whisper_model_path
        
        # Initialize OpenWakeWord
        self._init_wakeword()

        # Initialize Llama (Functional Gemma)
        self._init_llama()

        # Gather Tools
        self.tools_schema = [
            py_function_to_json(tool) 
            for name, tool in tools.__dict__.items() 
            if callable(tool) and getattr(tool, "__module__", "") == tools.__name__
        ]
        self.tools_map = {
            tool.__name__: tool
            for name, tool in tools.__dict__.items()
            if callable(tool) and getattr(tool, "__module__", "") == tools.__name__
        }

        # Audio Logic
        self.audio_queue = queue.Queue()
        self.pre_roll_buffer = collections.deque(maxlen=int(self.pre_roll_seconds / 0.08))
        
        # Threads
        self.record_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._inference_loop, daemon=True)

    def _init_wakeword(self):
        logging.info(f"Loading Wake Word Model: {self.ww_model_name}...")
        try:
            openwakeword.utils.download_models() 
            self.ww_model = Model(wakeword_models=[self.ww_model_name], inference_framework="onnx")
        except Exception as e:
            logging.error(f"Failed to init wake word: {e}")
            sys.exit(1)

    def _init_llama(self):
        logging.info(f"Loading Llama Model: {self.args.llm_model_path}...")
        
        # Load GGUF Model
        # n_gpu_layers=0 for Pi CPU only, modify if you have Vulkan
        self.llm = Llama(
            model_path=self.args.llm_model_path,
            n_ctx=2048,
            n_threads=4,  # Adjust based on Pi cores
            verbose=False,
            # Ensure chat format is detected or specified if needed
            # chat_format="chatml" # or "function_gemma" if supported/auto-detected
        )

    def start(self):
        self.record_thread.start()
        self.process_thread.start()
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logging.info("Stopping...")
        self.running = False

    def _audio_loop(self):
        # Same logic as original voice_assistant.py
        p = pyaudio.PyAudio()
        CHUNK = 1280
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        try:
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

        logging.info("Listening...")

        while self.running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                energy = np.sqrt(np.mean(audio_int16.astype(np.float32)**2))

                if self.debug_mode and not is_recording_command:
                    print(f"\rEnergy: {int(energy)}", end='', flush=True)

                if not is_recording_command:
                    self.pre_roll_buffer.append(data)
                    prediction = self.ww_model.predict(audio_int16)
                    # Assuming single model for simplicity
                    if prediction[self.ww_model_name] > 0.5:
                        logging.info("\n--- WAKE WORD DETECTED ---")
                        is_recording_command = True
                        command_frames.extend(self.pre_roll_buffer)
                        self.pre_roll_buffer.clear()
                        silence_start = None
                else:
                    command_frames.append(data)
                    if energy < self.vad_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        if (time.time() - silence_start) > self.silence_limit:
                            logging.info("Processing...")
                            self.audio_queue.put(b''.join(command_frames))
                            command_frames = []
                            is_recording_command = False
                            self.ww_model.reset()
                    else:
                        silence_start = None
            except Exception as e:
                logging.error(f"Audio loop error: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        p.terminate()

    def _run_whisper_cpp(self, wav_path):
        """Runs whisper.cpp via subprocess."""
        # Command: ./main -m models/ggml-base.en.bin -f file.wav -nt -otxt
        # We capture stdout instead of file output for speed
        cmd = [
            self.whisper_bin,
            "-m", self.whisper_model_path,
            "-f", wav_path,
            "--no-timestamps",  # -nt
            "--output-txt",     # Ensure text output
            "--no-prints"       # Don't print progress
        ]
        
        try:
            # whisper.cpp usually prints to stdout. 
            # We use capture_output=True to get it.
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            text = result.stdout.strip()
            # Fallback: whisper.cpp sometimes prints system info to stderr and text to stdout
            # If stdout is empty, check usage. 
            # Some versions might require -otxt which creates a file. 
            # Let's try basic stdout read.
            return text
        except subprocess.CalledProcessError as e:
            logging.error(f"Whisper failed: {e.stderr}")
            return ""

    def _inference_loop(self):
        while self.running:
            try:
                audio_bytes = self.audio_queue.get(timeout=1.0)
                
                # 1. Save to temp WAV for whisper.cpp
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_wav_path = f.name
                    with wave.open(temp_wav_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2) # 16-bit
                        wf.setframerate(16000)
                        wf.writeframes(audio_bytes)
                
                # 2. Run Whisper CPP
                user_text = self._run_whisper_cpp(temp_wav_path)
                os.remove(temp_wav_path)
                
                user_text = user_text.strip()
                if not user_text:
                    continue
                
                logging.info(f"User: {user_text}")

                # 3. Llama Inference (Function Calling)
                # Create messages compatible with Functional Gemma
                messages = [
                    {"role": "user", "content": user_text}
                ]
                
                # Use Llama-cpp built-in chat completion with tool support
                response = self.llm.create_chat_completion(
                    messages=messages,
                    tools=self.tools_schema,
                    tool_choice="auto",
                    max_tokens=256
                )
                
                choice = response["choices"][0]
                message = choice["message"]
                
                # 4. Parse and Execute
                if "tool_calls" in message and message["tool_calls"]:
                     self._handle_tool_calls(message["tool_calls"])
                else:
                    content = message.get("content", "")
                    logging.info(f"Assistant: {content}")

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Inference error: {e}")

    def _handle_tool_calls(self, tool_calls):
        for tool_call in tool_calls:
            try:
                # Handle both object and dict representation depending on library version
                if hasattr(tool_call, 'function'):
                     func_data = tool_call.function
                     func_name = func_data.name
                     args_str = func_data.arguments
                else:
                     func_data = tool_call["function"]
                     func_name = func_data["name"]
                     args_str = func_data["arguments"]

                if isinstance(args_str, str):
                    args = json.loads(args_str)
                else:
                    args = args_str

                if func_name in self.tools_map:
                    logging.info(f"Existing Tool: {func_name} with {args}")
                    
                    # Execute tool
                    result = self.tools_map[func_name](**args)
                    logging.info(f"Result: {result}")
                else:
                    logging.warning(f"Tool {func_name} not found.")

            except Exception as e:
                 logging.error(f"Failed to execute tool call {tool_call}: {e}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wake_word_model', default='hey_jarvis_v0.1')
    parser.add_argument('--device_index', type=int, default=-1)
    parser.add_argument('--vad_threshold', type=int, default=60)
    parser.add_argument('--silence_limit', type=float, default=1.2)
    parser.add_argument('--pre_roll_seconds', type=float, default=1.0)
    parser.add_argument('--debug_mode', action='store_true')
    
    # New arguments
    parser.add_argument('--whisper_bin_path', required=True, help="Path to whisper.cpp 'main' executable")
    parser.add_argument('--whisper_model_path', required=True, help="Path to whisper.cpp ggml model")
    parser.add_argument('--llm_model_path', required=True, help="Path to GGUF model")
    # No tokenizer path needed anymore
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    assistant = RaspiAssistant(args)
    assistant.start()
