import os
import pandas as pd
from datasets import Dataset
import json

def check_file_exists(file_path, file_type):
    """Check if a file exists and is readable."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_type} file not found at: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Cannot read {file_type} file at: {file_path}")

def load_character_data(jsonl_path):
    """Load character information from JSONL file."""
    check_file_exists(jsonl_path, "Character JSONL")
    characters = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            char_data = json.loads(line)
            title = char_data['title']
            characters[title] = {
                'role': char_data['role'],
                'description': char_data['desc']
            }
    return characters

def load_shakespeare_data(csv_path, jsonl_path):
    # Load character information
    print("Loading character data...")
    characters = load_character_data(jsonl_path)
    
    # Load the CSV file
    print("Loading play data...")
    check_file_exists(csv_path, "CSV")
    df = pd.read_csv(csv_path)
    
    # Create conversation pairs with character information
    conversations = []
    current_play = None
    current_act_scene = None
    current_conversation = []
    
    for _, row in df.iterrows():
        if pd.isna(row['Play']) or pd.isna(row['ActSceneLine']) or pd.isna(row['Player']) or pd.isna(row['PlayerLine']):
            continue
            
        # Start new conversation if play or act/scene changes
        if row['Play'] != current_play or row['ActSceneLine'] != current_act_scene:
            if current_conversation:
                conversations.append("\n".join(current_conversation))
            current_conversation = []
            current_play = row['Play']
            current_act_scene = row['ActSceneLine']
            
            # Add character information at the start of each scene
            if current_play in characters:
                char_info = characters[current_play]
                current_conversation.append(f"Play: {current_play}")
                current_conversation.append(f"Role: {char_info['role']}")
                current_conversation.append(f"Description: {char_info['description']}")
                current_conversation.append("---")
        
        # Add the line to current conversation
        current_conversation.append(f"{row['Player']}: {row['PlayerLine']}")
    
    # Add the last conversation
    if current_conversation:
        conversations.append("\n".join(current_conversation))
    
    # Create dataset from conversations
    dataset = Dataset.from_dict({"text": conversations})
    
    # Split into train and validation sets
    train_val_split = dataset.train_test_split(test_size=0.1)
    
    return train_val_split

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths relative to the script directory
    csv_path = os.path.join(script_dir, "archive\Shakespeare_data.csv")
    jsonl_path = os.path.join(script_dir, "characters.jsonl")
    
    print("Processing Shakespeare data...")
    print(f"Looking for CSV file at: {csv_path}")
    print(f"Looking for JSONL file at: {jsonl_path}")
    
    try:
        dataset = load_shakespeare_data(csv_path, jsonl_path)
        print(f"Training set size: {len(dataset['train'])}")
        print(f"Validation set size: {len(dataset['test'])}")
        
        # Save the processed dataset
        dataset.save_to_disk("shakespeare_processed")
        print("Dataset saved successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease make sure both files exist in the correct location.")
        print("The CSV file should be in the archive folder")
        print("The JSONL file should be named 'characters.jsonl'")
    except Exception as e:
        print(f"An error occurred: {e}") 