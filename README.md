# Shakespeare Chatbot

This project fine-tunes a DistilGPT2 model on Shakespeare's plays to create a chatbot that can generate text in Shakespeare's style.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Kaggle credentials:
   - Go to your Kaggle account settings
   - Create a new API token
   - Place the `kaggle.json` file in `~/.kaggle/` (Linux/Mac) or `C:\Users\<Windows-username>\.kaggle\` (Windows)

## Usage

1. Process the Shakespeare dataset:
```bash
python process_data.py
```

2. Train the model:
```bash
python train.py
```

The training process will:
- Download the Shakespeare plays dataset
- Process and tokenize the text
- Fine-tune the DistilGPT2 model
- Save the trained model to `./shakespeare_model_final`

## Model Details

- Base Model: DistilGPT2
- Training Data: Shakespeare's plays
- Context Length: 512 tokens
- Training Epochs: 3
- Batch Size: 4

## Monitoring

The training progress can be monitored through Weights & Biases (wandb). Make sure to:
1. Create a wandb account
2. Login using `wandb login`
3. The training metrics will be automatically logged to your wandb dashboard

## Output

The trained model will be saved in the `shakespeare_model_final` directory, which includes:
- The fine-tuned model weights
- The tokenizer configuration
- Model configuration files # ShakespeareChatBot
