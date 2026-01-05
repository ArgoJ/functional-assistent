import json
import glob
import os
import torch

from huggingface_hub import login
from trl import SFTConfig, SFTTrainer
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema

import tools
from checker import check_success_rate



base_model = "google/functiongemma-270m-it"
learning_rate = 5e-5


TOOLS = [get_json_schema(tool) for name, tool in tools.__dict__.items() if callable(tool) and getattr(tool, "__module__", "") == tools.__name__]
DEFAULT_SYSTEM_MSG = (
    "Du bist ein hilfreicher Assistent mit Zugriff auf spezifische Werkzeuge.\n",
    "1. Prüfe zuerst, ob eine spezifische Funktion (wie Wetter, Musik, Alarm) die Anfrage lösen kann.\n",
    "2. Wenn keine spezifische Funktion passt, nutze die Websuche ('search_web'), um Informationen zu finden.\n",
    "3. Nutze nur dann reinen Text, wenn gar keine Funktion passt (z.B. bei Begrüßungen). Antworte immer im validen Funktionsaufruf-Format oder auf Deutsch.\n")

def create_conversation(sample, tool_names=None):
    tool_name = sample.get("tool_name")
    
    if tool_names and isinstance(tool_name, int):
        tool_name = tool_names[tool_name]

    if tool_name and tool_name != "null":
        assistant_message = {
            "role": "assistant",
            "tool_calls": [{
                "type": "function", 
                "function": {
                    "name": tool_name, 
                    "arguments": sample["tool_arguments"]
                }
            }]
        }
    else:
        response_text = sample.get("response", "Ich kann dir dabei leider nicht helfen.")
        assistant_message = {
            "role": "assistant",
            "content": response_text
        }

    return {
        "messages": [
            {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
            {"role": "user", "content": sample["user_content"]},
            assistant_message,
        ],
        "tools": TOOLS
    }


# Prepare dataset
loaded_json = []
for file_path in glob.glob(os.path.abspath("data/*.json")):
    with open(file_path, "r") as f:
        loaded_json.extend(json.load(f))

dataset = Dataset.from_list(loaded_json)

# Cast tool_name to ClassLabel for stratified splitting
tool_names = sorted(list(set(item["tool_name"] for item in loaded_json)))
dataset = dataset.cast_column("tool_name", ClassLabel(names=tool_names))

# Map without removing columns to allow stratification on tool_name
original_columns = dataset.column_names
dataset = dataset.map(create_conversation, fn_kwargs={"tool_names": tool_names}, batched=False)

dataset = dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column="tool_name")

# Remove original columns after splitting
dataset["train"] = dataset["train"].remove_columns(original_columns)
dataset["test"] = dataset["test"].remove_columns(original_columns)


# Load model and tokenizer 
login(token=os.getenv("HF_TOKEN"))

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

print(f"Device: {model.device}")
print(f"DType: {model.dtype}")

print("\n--- Initial check before training ---")
check_success_rate(dataset["test"], model, tokenizer, TOOLS)


# Define training arguments
torch_dtype = model.dtype
args = SFTConfig(
    output_dir="./functiongemma-tool-calling-sft",              # directory to save and repository id
    max_length=1024,                         # max sequence length for model and packing of the dataset
    packing=False,                          # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=64,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_checkpointing=False,           # Caching is incompatible with gradient checkpointing
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                        # log every step
    #save_strategy="epoch",                  # save checkpoint every epoch
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    learning_rate=learning_rate,            # learning rate
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,  # use bfloat16 precision
    lr_scheduler_type="constant",            # use constant learning rate scheduler
    push_to_hub=False,                        # push model to hub
    report_to="tensorboard",                 # report metrics to tensorboard
)


# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

print("\n--- Check after training ---")
check_success_rate(dataset["test"], model, tokenizer, TOOLS)

# Save the final model again to the Hugging Face Hub
# trainer.save_model()