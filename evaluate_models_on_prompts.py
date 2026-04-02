import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# for GPTQ loading
from gptqmodel import GPTQModel, QuantizeConfig

# ——— File‐write setup ———
LOG_PATH = "evaluation.log"
# ensure parent dir exists (if needed)
os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
log_file = open(LOG_PATH, "w", encoding="utf-8")

def write_log(line: str):
    """Write a line to the log file with a timestamp."""
    print(line)
    log_file.write(f"{line}\n")
    log_file.flush()

# Define your model names here
MODEL_NAMES = {
    "distilgpt2": "distilbert/distilgpt2",
    "mistral_7b": "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",  # GPTQ-quantized
    "phi_2":      "microsoft/phi-2",
    "llama_3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma_2b":   "google/gemma-2b-it"
}

# configure quantization defaults
def get_quant_config(device_str: str):
    return QuantizeConfig(
        bits=4,
        group_size=128,
        device=device_str
    )

def load_model(model_id: str):
    write_log(f"→ Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if model_id.endswith("GPTQ"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        quant_config = get_quant_config(str(device))
        model = GPTQModel.load(model_id, quant_config)
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            use_auth_token=True
        )

    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def evaluate_model(model_key: str, prompts: list[str], max_length: int = 150):
    try:
        tokenizer, model = load_model(MODEL_NAMES[model_key])
    except Exception as e:
        write_log(f"❗ Skipping {model_key} due to error: {e}")
        return

    model.eval()
    for i, prompt in enumerate(prompts, 1):
        write_log(f"--- Prompt {i} ---")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        write_log(text)

if __name__ == "__main__":
    from shakespeare_eval_prompts_ordered import prompts

    write_log("===== Questions =====")
    for i, p in enumerate(prompts, 1):
        write_log(f"Question {i}: {p}")

    model_keys = list(MODEL_NAMES.keys())
    if not torch.cuda.is_available():
        write_log("⚠️ No CUDA — skipping GPTQ-quantized models.")
        model_keys = [k for k in model_keys if not MODEL_NAMES[k].endswith("GPTQ")]

    write_log("===== Evaluation =====")
    for key in model_keys:
        write_log(f"=== Evaluating {key} ===")
        evaluate_model(key, prompts)

    log_file.close()
    print(f"Done! All output written to {LOG_PATH}")
