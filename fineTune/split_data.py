import random
import os

# Simple script to split a JSONL file into train and validation sets
def split_jsonl(filename, train_ratio=0.85):
    # 1. Read all lines
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 2. Shuffle to avoid bias (e.g., if all "Introduction" sections are at the top)
    random.shuffle(lines)
    
    # 3. Calculate split index
    split_idx = int(len(lines) * train_ratio)
    
    # 4. Slice
    train_data = lines[:split_idx]
    valid_data = lines[split_idx:]
    
    # 5. Save
    dirname = filename.split(".")[0]
    os.makedirs(dirname, exist_ok=True)
    with open(f'{dirname}/train.jsonl', 'w') as f:
        f.writelines(train_data)
        
    with open(f'{dirname}/valid.jsonl', 'w') as f:
        f.writelines(valid_data)
        
    print(f"Done! {len(train_data)} training examples, {len(valid_data)} validation examples.")

if __name__ == "__main__":
    split_jsonl("data/eli5_2048.jsonl")
    split_jsonl("data/executive_2048.jsonl")
    split_jsonl("data/intuitive_2048.jsonl")
    
