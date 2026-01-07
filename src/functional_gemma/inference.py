import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema
import os
import glob
import tools  # Deine Tools

# 1. Pfad zum neuesten Checkpoint finden
output_dir = "./functiongemma-tool-calling-sft"
checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
if not checkpoints:
    print("Keine Checkpoints gefunden! Nutze Basis-Modell?")
    checkpoint_path = "google/functiongemma-270m-it"
else:
    # Sortiere nach Nummer (z.B. checkpoint-160, checkpoint-320)
    checkpoint_path = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    print(f"Lade Modell aus: {checkpoint_path}")

# 2. Modell & Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    device_map="auto",
    torch_dtype="auto"
)
model.eval()

TOOLS = [get_json_schema(tool) for name, tool in tools.__dict__.items() 
         if callable(tool) and getattr(tool, "__module__", "") == tools.__name__]

# 4. Inference Funktion
def ask_model(query):
    
    messages = [
        {"role": "user", "content": query}
        # Kein System-Prompt nötig, macht das Template oft selbst, oder wir geben den Default mit
    ]
    
    # Prompt bauen
    inputs = tokenizer.apply_chat_template(
        messages, 
        tools=TOOLS, 
        add_generation_prompt=True, # Wichtig: Signalisiert "Model ist dran"
        return_dict=True, 
        return_tensors="pt"
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generieren
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128,
        do_sample=False # Deterministisch ist besser für Tools
    )
    
    # Nur den neuen Teil decoden
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    
    return generated_text

# 5. Testlauf
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