import os
import glob
import sys
from transformers import AutoTokenizer
from transformers.utils import get_json_schema
import tools  # Custom tools

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed.")
    print("Please install it with: pip install llama-cpp-python")
    sys.exit(1)

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# 1. Load Tokenizer (from original or checkpoint for chat template)
# We need the tokenizer mostly for Apply Chat Template
# We can load it from the base model if necessary, but checkpoint is safer
output_dir = os.path.join(project_root, "functiongemma-tool-calling-sft")
checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
if checkpoints:
    checkpoint_path = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    print(f"Loading tokenizer from: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
else:
    print("Warning: No checkpoint found to load tokenizer config. Using base model.")
    tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")

# 2. Find GGUF Model
gguf_files = glob.glob(os.path.join(project_root, "*.gguf"))
if not gguf_files:
    print(f"No .gguf files found in {project_root}")
    print("Please run export_gguf.py first.")
    sys.exit(1)

# Pick the latest modified gguf file
gguf_model_path = max(gguf_files, key=os.path.getmtime)
print(f"Loading GGUF model: {gguf_model_path}")

# Load Llama Model
# n_ctx should be enough for prompt + generation
llm = Llama(
    model_path=gguf_model_path,
    n_ctx=2048,
    n_threads=4,  # Adjust based on CPU
    verbose=False
)

TOOLS = [get_json_schema(tool) for name, tool in tools.__dict__.items() 
         if callable(tool) and getattr(tool, "__module__", "") == tools.__name__]

def ask_model(query):
    messages = [
        {"role": "user", "content": query}
    ]
    
    # Use tokenizer to apply the chat template with tools
    # format_chat_template logic from finetuning/inference
    prompt = tokenizer.apply_chat_template(
        messages, 
        tools=TOOLS, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # Run Inference
    # stop=["<end_of_turn>"] is important for Gemma
    output = llm(
        prompt,
        max_tokens=128,
        stop=["<end_of_turn>", "<eos>"],
        echo=False
    )
    
    return output['choices'][0]['text']

if __name__ == "__main__":
    test_queries = [
        "Turn on the lights in the kitchen.",
        "What is the weather in Berlin tomorrow?",
        "Play specific Rock music."
    ]
    
    print("-" * 50)
    for q in test_queries:
        print(f"User: {q}")
        response = ask_model(q)
        print(f"Model: {response}")
        print("-" * 50)
