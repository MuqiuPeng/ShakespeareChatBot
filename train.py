import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import json
import datasets
import wandb
import sys

def prepare_dataset(tokenizer, dataset, max_length=512):
    def tokenize_function(examples):
        texts = []
        for title, role, desc in zip(examples["title"], examples["role"], examples["desc"]):
            question = f"Who is {role}?"
            answer = f"Play: {title}\nRole: {role}\nDescription: {desc}"
            text = f"Human: {question}\nShakespeare: {answer}\n---\n"
            texts.append(text)
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )
    tokenized_train = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_test = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["test"].column_names
    )
    return datasets.DatasetDict({
        "train": tokenized_train,
        "test": tokenized_test
    })

def main():
    wandb.init(project="shakespeare-chatbot")

    # Model and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    print("LoRA adapters added. Only adapters will be trained.")
    print(model.print_trainable_parameters())

    # Check trainable parameters
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        print("ERROR: No trainable parameters found! This model may not support LoRA/PEFT training.")
        sys.exit(1)

    # Device info
    print("PyTorch version:", torch.__version__)
    print("CUDA available?:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  → Device count:", torch.cuda.device_count())
        print("  → Current device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Special tokens
    special_tokens = {
        "pad_token": "<PAD>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "additional_special_tokens": [
            "Human:", "Shakespeare:", "Play:", "Role:", "Description:", "---"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Load and prepare data
    with open("characters.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    dataset_dict = {
        "title": [item["title"] for item in data],
        "role": [item["role"] for item in data],
        "desc": [item["desc"] for item in data]
    }
    dataset = datasets.Dataset.from_dict(dataset_dict)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    tokenized_dataset = prepare_dataset(tokenizer, dataset, max_length=512)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./shakespeare_model",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
        fp16=True,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        max_grad_norm=1.0,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model("./shakespeare_model_final")
    tokenizer.save_pretrained("./shakespeare_model_final")

if __name__ == "__main__":
    main()
