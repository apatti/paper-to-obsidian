import os
import re
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# We use Qwen2-VL-2B (Quantized). It's tiny (~1.5GB) but SOTA for charts/OCR.
VISION_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

def caption_images_in_markdown(md_content, base_path):
    """
    Finds all ![img](path) links, runs a VLM on them, and inserts the description.
    """
    # Find all image links: ![alt](path)
    # We use a regex that captures the path
    image_links = re.findall(r'!\[.*?\]\((.*?)\)', md_content)
    
    if not image_links:
        return md_content

    print(f"Found {len(image_links)} images. Waking up Vision Model...")
    
    # Load Model (Only stays in RAM for this function)
    model, processor = load(VISION_MODEL)
    config = load_config(VISION_MODEL)
    
    new_content = md_content

    for rel_path in image_links:
        # Construct full path (Obsidian uses relative paths, Python needs absolute)
        # Warning: You might need to adjust this join depending on your folder structure
        full_path = os.path.join(base_path, rel_path)
        
        if not os.path.exists(full_path):
            print(f"Image not found: {full_path}")
            continue

        print(f"Captioning: {rel_path}...", end="\r")
        
        # Prompt for Research Papers
        prompt = "Describe this image in detail. If it's a chart, read the data. If it's a diagram, explain the flow."
        
        formatted_prompt = apply_chat_template(
            processor, config, prompt, num_images=1
        )

        # Generate Caption
        output = generate(
            model, 
            processor, 
            formatted_prompt, 
            [full_path], 
            max_tokens=500, 
            verbose=False, 
            repeat_penalty=1.1 # <--- 1.1 or 1.2 stops loops
        )
        
        # Extract text from GenerationResult object
        caption_text = output.text.strip().replace("\n", " ")
        
        # Inject Caption into Markdown
        # We turn: ![img](path)
        # Into:  ![img](path)
        #        > **AI Vision:** *A diagram showing the Transformer architecture...*
        caption_block = f"({rel_path})\n> [!INFO] AI Vision\n> *{caption_text}*\n"

        new_content = new_content.replace(f"({rel_path})", caption_block)

    print(f"   Captioned {len(image_links)} images.")
    return new_content