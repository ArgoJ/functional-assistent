# %% Imports
import json
import glob
import os
import torch

from huggingface_hub import login
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema

import tools
from checker import check_success_rate
from plot import plot_training_loss

class MyCompletionOnlyDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, response_template, *args, **kwargs):
        super().__init__(pad_token_id=tokenizer.pad_token_id, *args, **kwargs)
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        
        for i in range(len(labels)):
            input_ids = batch["input_ids"][i]
            start_idx = -1
            
            # Suche nach der response_template Sequenz
            token_len = len(self.response_template_ids)
            for j in range(len(input_ids) - token_len + 1):
                if input_ids[j:j+token_len].tolist() == self.response_template_ids:
                    start_idx = j + token_len
                    break
            
            if start_idx != -1:
                # Maskiere alles VOR dem Start der Antwort
                labels[i, :start_idx] = -100
            else:
                print(f"WARNUNG: Response Template nicht gefunden in Beispiel {i}!")
                # Optional: Zeige Tokens an, um zu debuggen
                # print(tokenizer.decode(input_ids))
                
        batch["labels"] = labels
                
        return batch

# %% Base Configuration
base_model = "google/functiongemma-270m-it"
learning_rate = 2e-5
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
data_path = os.path.join(project_root, "data")
print(f"Using data from: {data_path}")


TOOLS = [get_json_schema(tool) for name, tool in tools.__dict__.items() if callable(tool) and getattr(tool, "__module__", "") == tools.__name__]
DEFAULT_SYSTEM_MSG = (
    "You are a helpful assistant. "
    "Use the provided tools to answer questions. "
    "If no tool fits, use 'search_web'.")
# DEFAULT_SYSTEM_MSG = (
#     "Du bist ein hilfreicher Assistent. "
#     "Nutze die bereitgestellten Tools, um Fragen zu beantworten. "
#     "Wenn kein Tool passt, nutze 'search_web'.")

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

# %% Prepare dataset
loaded_json = []
for file_path in glob.glob(os.path.join(data_path, "*.json")):
    if "negative" in file_path:
        continue
    with open(file_path, "r") as f:
        loaded_json.extend(json.load(f))

# Ensure tool_name is never None for ClassLabel and sorting
for item in loaded_json:
    if item["tool_name"] is None:
        item["tool_name"] = "null"

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

# %% Load model and tokenizer
login(token=os.getenv("HF_TOKEN"))

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.padding_side = "right" # Wichtig für Gemma
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)


#  %% Print debug information
print("--- Model and Dataset Info ---")
print(f"Device: {model.device}")
print(f"DType: {model.dtype}")

# Apply chat template to create a single text column for training
# This ensures tools are rendered into the prompt string
def format_chat_template(row):
    return {
        "text": tokenizer.apply_chat_template(
            row["messages"], 
            tools=row["tools"], 
            add_generation_prompt=False, 
            tokenize=False
        )
    }

print("Applying chat template to dataset...")
dataset = dataset.map(format_chat_template)

print("--- dataset input ---")
print(json.dumps(dataset["train"][0]["messages"], indent=2))
print("--- Formatted prompt (Training Data) ---")
print(dataset["train"][0]["text"])


# %%
print("\n--- Initial check before training ---")
check_success_rate(dataset["test"], model, tokenizer, TOOLS)

# %% Define training arguments
torch_dtype = model.dtype
args = SFTConfig(
    output_dir="./functiongemma-tool-calling-sft",
    dataset_text_field="text",              # use formated row 
    max_length=1024,
    packing=False,                          # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=8,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_checkpointing=False,           # Caching is incompatible with gradient checkpointing
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                        # log every step
    #save_strategy="epoch",                  # save checkpoint every epoch
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    learning_rate=learning_rate,            # learning rate
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,  # use bfloat16 precision
    lr_scheduler_type="cosine",            # use constant learning rate scheduler
    push_to_hub=False,                        # push model to hub
    report_to="tensorboard",                 # report metrics to tensorboard
)

# NEU: Data Collator definieren
response_template = "<start_of_turn>model"
data_collator = MyCompletionOnlyDataCollator(
    tokenizer=tokenizer,
    response_template=response_template
)

# %% Train the model
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# %% Plot training loss
plot_training_loss(trainer)

# %% Final evaluation
print("\n--- Check after training ---")
check_success_rate(dataset["test"], model, tokenizer, TOOLS)