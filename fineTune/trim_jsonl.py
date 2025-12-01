import json
from pathlib import Path
from mlx_lm.tokenizer_utils import load_tokenizer

# 3. THE SAFETY LIMIT (2048 is stable for M2 Pro)
MAX_TOKENS = 2048

# 4. THE MODEL (Needed to count tokens correctly)
MODEL_PATH = "mlx-community/Llama-3.2-3B-Instruct-4bit" 

def clean_data(input_file, output_file):
    print(f"Loading tokenizer from {MODEL_PATH}...")
    # We load the tokenizer so we count EXACTLY like the model does
    tokenizer = load_tokenizer(Path(MODEL_PATH))
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        skipped = 0
        trimmed = 0
        valid = 0
        
        print(f"Scanning {input_file}...")
        
        for i, line in enumerate(fin):
            try:
                if not line.strip(): continue
                entry = json.loads(line)
                
                # Check format
                if "messages" not in entry: 
                    continue

                user_msg = entry["messages"][0]["content"]
                asst_msg = entry["messages"][1]["content"]
                
                # 1. CHECK TOTAL LENGTH
                # We apply the chat template because special tokens (like <|eot_id|>) count towards the limit!
                full_text = tokenizer.apply_chat_template(entry["messages"], tokenize=False)
                tokens = tokenizer.encode(full_text)
                
                if len(tokens) <= MAX_TOKENS:
                    # Case A: It fits perfectly. Save it.
                    fout.write(line)
                    valid += 1
                else:
                    # Case B: It's too big. We need to trim.
                    # CRITICAL RULE: Never trim the Output (Assistant). Only trim the Input (User).
                    
                    # Calculate how much space the Answer takes
                    asst_tokens = tokenizer.encode(asst_msg)
                    len_asst = len(asst_tokens)
                    
                    # Calculate remaining budget for the User
                    # Total Limit - Answer - 100 tokens buffer (for system headers/special chars)
                    user_budget = MAX_TOKENS - len_asst - 100
                    
                    if user_budget < 50:
                        # If the Answer alone is huge (e.g. 2000 tokens), there is no room for a question.
                        # We must skip this data point.
                        skipped += 1
                        continue
                        
                    # Slice the User's input to fit the budget
                    user_tokens = tokenizer.encode(user_msg)
                    trimmed_user_tokens = user_tokens[:user_budget]
                    
                    # Decode back to text and add a marker
                    trimmed_user_text = tokenizer.decode(trimmed_user_tokens) + "\n...[Truncated]"
                    
                    # Create the new safe entry
                    new_entry = {
                        "messages": [
                            {"role": "user", "content": trimmed_user_text},
                            {"role": "assistant", "content": asst_msg}
                        ]
                    }
                    fout.write(json.dumps(new_entry) + "\n")
                    trimmed += 1
                    
            except Exception as e:
                print(f"Error on line {i}: {e}")
                skipped += 1

    print("="*40)
    print(f"Processing Complete!")
    print(f"Kept (As-is):   {valid}")
    print(f"Trimmed Input:  {trimmed}")
    print(f"Skipped:        {skipped}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    inputs = ["eli5", "executive", "intuitive"]
    for input in inputs:
        input_file = f"data/{input}.jsonl"
        output_file = f"data/{input}_2048.jsonl"
        clean_data(input_file, output_file)