import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path="./shakespeare_model_final"):
    """Load the trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def generate_response(model, tokenizer, user_input, chat_history_ids=None, max_new_tokens=200):
    """Generate a response using the trained model."""
    # Format the input with the chat template
    prompt = f"Human: {user_input}\nAssistant:"
    
    # Encode the new user input
    new_input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Append the new input tokens to the chat history
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids
    
    # Generate response
    with torch.no_grad():
        chat_history_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2
        )
    
    # Decode and return the response
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

def main():
    print("Loading Shakespeare chatbot model...")
    model, tokenizer = load_model()
    print("Model loaded successfully!")
    print("\nWelcome to the Shakespeare Chatbot!")
    print("Type 'quit' to exit the chat.")
    print("-" * 50)
    
    chat_history_ids = None
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nFarewell, good friend!")
            break
        
        # Generate response
        response, chat_history_ids = generate_response(model, tokenizer, user_input, chat_history_ids)
        
        # Print response
        print("\nShakespeare:", response)

if __name__ == "__main__":
    main() 