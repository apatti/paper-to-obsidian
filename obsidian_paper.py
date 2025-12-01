import re
from mlx_lm import load, generate
import pymupdf4llm
from typing import List, Dict, Tuple
import os
import pathlib
import sys
import shutil
from dotenv import load_dotenv
from utils.vision import caption_images_in_markdown

load_dotenv()

OBSIDIAN_VAULT_PATH = os.environ.get("OBSIDIAN_VAULT_PATH")
if not OBSIDIAN_VAULT_PATH:
    raise EnvironmentError("OBSIDIAN_VAULT_PATH is not set in .env file")
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

ADAPTERS = {
    "eli5": "adapters/eli5_final/",
    "executive": "adapters/executive_final/",
    "intuitive": "adapters/intuitive_final/"
}

STYLE_CONFIG = {
    "executive": {
        "callout": "> [!SUMMARY] Executive Verdict",
        "prompt": """
        Give me an executive summary:
        CRITICAL: If you mention other famous models (like GPT-4, BERT) or concepts or companies, wrap them in double brackets like [[BERT]]
        """
    },
    "intuitive": {
        "callout": "> [!EXAMPLE] Engineering Logic",
        "prompt": """
        Explain the intuition behind this:
        CRITICAL: Wrap technical concepts (like [[Backpropagation]], [[Transformer]], [[ReLU]]) in double brackets to link them to my knowledge base.
        """
    },
    "eli5": {
        "callout": "> [!TIP] ELI5 Analogy",
        "prompt": """
        Explain this like I'm 5:
        """
    }
}

paper_full_text = ""


def clear_paper_file_name(file_name: str) -> str:
    return "".join(c for c in file_name if c.isalnum() or c.isspace()).rstrip()

r'''def extract_paper_title(paper_text: str) -> Dict[str, str]:
    #print(paper_text)
    title_pattern = r"#?\s+(.+)abstract"
    match = re.search(title_pattern, paper_text, re.IGNORECASE | re.DOTALL)
    print(match)
    if match:
        title = match.group(1)
        print(title)
        return title
    return None
    '''

def extract_paper_title(paper_text: str, paper_name:str) -> str:
    # 1. Capture everything from start until "Abstract"
    # We use specific keywords to stop early if Abstract isn't found immediately
    title_pattern = r"^#?\s*(.+?)(?:\n\n\n|\n\n.*?abstract|author|By\s)"
    
    # Fallback: The original regex is good, just adding non-greedy (+?) to be safer
    match = re.search(r"#?\s+(.+?)abstract", paper_text, re.IGNORECASE | re.DOTALL)
    
    if match:
        # Get the big block of text before abstract
        header_block = match.group(1).strip()
        
        # 2. Split by double newlines to separate "Title Block" from "Author Block"
        # In Markdown, paragraphs are separated by \n\n. 
        # Titles/Subtitles are usually one paragraph. Authors are the next.
        title_chunk = header_block.split('\n\n')[0]
        
        # 3. Clean up the Title Chunk
        # Remove markdown headers (#, **)
        clean_title = title_chunk.replace('#', '').replace('*', '').strip()
        
        # Merge multi-line titles (handling subtitles)
        # If it was "Title \n Subtitle", it becomes "Title: Subtitle" or "Title Subtitle"
        if '\n' in clean_title:
            # Replace newline with a space (or ": " if you prefer)
            clean_title = " ".join([line.strip() for line in clean_title.split('\n')])
            
        return clean_title

    return "Unknown Title"

def parse_pdf_sections_robust(pdf_path: str, image_subfolder: str, image_path: str) -> List[Tuple[str, str]]:
    # 1. Convert PDF to Markdown
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        image_path=image_path,
        write_images=True,
        image_format="png"
    )

    # 2. Fix Image Paths
    abs_path_prefix = str(pathlib.Path(image_path).absolute())
    md_text = md_text.replace(abs_path_prefix, image_subfolder)

    # 2.1 caption images
    md_text = caption_images_in_markdown(md_text, OBSIDIAN_VAULT_PATH)

    global paper_full_text
    paper_full_text = md_text

    # 3. Regex for standard Markdown headers (#)
    header_pattern = re.compile(r'^(#+)\s+(.*)$')

    sections = []
    current_top_level_header = "Abstract" # Default bucket
    buffer_text = []

    lines = md_text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue # Skip empty lines to keep buffer clean

        # --- DETECT POTENTIAL HEADER ---
        is_header_candidate = False
        clean_title = ""

        # Case A: Standard Markdown Header (# Title)
        match_hash = header_pattern.match(line)
        if match_hash:
            clean_title = match_hash.group(2).strip().replace('*', '')
            is_header_candidate = True
        
        # Case B: Bold Header (**1** **Introduction**)
        # Check if line starts/ends with ** and is short enough to be a title (< 100 chars)
        elif line.startswith('**') and len(line) < 100:
            clean_title = line.replace('*', '').strip()
            is_header_candidate = True

        # --- FILTER LOGIC ---
        is_top_level = False
        
        if is_header_candidate:
            # Check 1: Specific Keywords
            if any(x == clean_title.lower() for x in ["abstract", "references", "acknowledgements", "bibliography"]):
                is_top_level = True
            
            # Check 2: Numbering (1, 2, A, B)
            # Reject "5.1" (dot in middle) but accept "1", "1.", "A", "A."
            first_word = clean_title.split(' ')[0]
            if re.match(r'^(\d+|[A-Z])\.?$', first_word):
                 is_top_level = True

        if is_top_level:
            # FLUSH PREVIOUS SECTION
            if buffer_text:
                joined_text = "\n".join(buffer_text).strip()
                if len(joined_text) > 50: # Filter noise
                    sections.append((current_top_level_header, joined_text))
            
            # START NEW SECTION
            current_top_level_header = clean_title
            buffer_text = []
        else:
            # It's body text (or a subsection like 5.1). Append line.
            buffer_text.append(line)

    # Append final section
    if buffer_text:
        joined_text = "\n".join(buffer_text).strip()
        if len(joined_text) > 50:
            sections.append((current_top_level_header, joined_text))

    return sections

def parse_pdf_sections(pdf_path: str, image_subfolder: str, image_path: str) -> List[Tuple[str, str]]:
    
    # 1. Convert PDF to Markdown
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        image_path=image_path,
        write_images=True,
        image_format="png"
    )

    # 2. Fix Image Paths for Obsidian
    abs_path_prefix = str(pathlib.Path(image_path).absolute())
    md_text = md_text.replace(abs_path_prefix, image_subfolder)
    
    # 3. Regex to catch ANY header line (1 to 6 hashes)
    header_pattern = re.compile(r'^(#+)\s+(.*)$', re.MULTILINE)
    
    sections = []
    current_top_level_title = "Abstract"
    buffer_text = []
    
    # Split by lines to process sequentially
    lines = md_text.split('\n')
    
    for line in lines:
        match = header_pattern.match(line)
        if match:
            # We found a header line!
            title_text = match.group(2).strip()
            
            # Clean formatting (remove **bold** wrappers often added by parsers)
            clean_title = title_text.replace('*', '').strip()
            
            # 4. FILTER LOGIC: Is this a Top-Level Section?
            is_top_level = False
            
            # Criteria A: Starts with "1 ", "1. ", "A ", "A. " (Single digit/letter)
            # We verify there is NO dot in the middle (rejects 5.1, 5.1.2)
            first_word = clean_title.split(' ')[0]
            # Regex: Starts with digit or single letter, optionally followed by dot
            if re.match(r'^(\d+|[A-Z])\.?$', first_word):
                 is_top_level = True
            
            # Criteria B: Specific Keywords
            if any(x == clean_title.lower() for x in ["abstract", "references", "acknowledgements", "bibliography"]):
                is_top_level = True

            if is_top_level:
                # -- FLUSH PREVIOUS SECTION --
                if buffer_text:
                    joined_text = "\n".join(buffer_text).strip()
                    if len(joined_text) > 50: # Filter empty/tiny sections
                        sections.append((current_top_level_title, joined_text))
                
                # -- START NEW SECTION --
                current_top_level_title = clean_title
                buffer_text = [] # Reset buffer
            else:
                # It is a subsection (e.g. 5.1) or a minor header. 
                # Treat it as bold text inside the current section.
                buffer_text.append(f"\n**{clean_title}**\n") 
        else:
            # Normal text line
            buffer_text.append(line)

    # Append the very last section
    if buffer_text:
        sections.append((current_top_level_title, "\n".join(buffer_text).strip()))

    return sections

def extract_concepts(full_text):
    """
    Special step: Ask the model to generate a list of Tags/Topics for the Graph.
    """
    # Use Executive brain for this as it's good at extraction
    model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTERS['executive'])
    
    prompt = f"""
    Analyze this text. List the Top 5 key technical concepts, architectures, or algorithms mentioned.
    Format them as a comma-separated list wrapped in brackets.
    Example: [[Transformer]], [[Attention Mechanism]], [[Google]], [[NLP]], [[Optimization]]
    
    Text:
    {full_text[:6000]}
    """
    
    messages = [{"role": "user", "content": prompt}]
    prompt_fmt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt_fmt, max_tokens=100, verbose=False)
    
    # Simple cleanup to ensure they look like links
    return response.strip()

def extract_visuals_only(content: str):
    lines = content.split('\n')
    visuals = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Found an image?
        if line.startswith('!['):
            block = [line]
            # Check if the NEXT lines are part of the Vision Caption (Blockquotes)
            # i.e., look for lines starting with '>' immediately following the image
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith('>'):
                block.append(lines[j])
                j += 1
            
            visuals.append("\n".join(block))
            i = j # Skip forward
        else:
            i += 1
    return visuals

def main():
    global paper_full_text
    if(len(sys.argv) != 2):
        print("Usage: python obsidian_paper.py <paper_name>")
        return
    pdf_path = sys.argv[1]
    paper_name = os.path.basename(pdf_path).replace(".pdf", "")
    safe_name = "".join(c for c in paper_name if c.isalnum() or c.isspace()).rstrip()

    output_file = os.path.join(OBSIDIAN_VAULT_PATH, f"{safe_name}.md")

    image_subfolder = f"assets/{safe_name}"
    full_image_path = os.path.join(OBSIDIAN_VAULT_PATH, image_subfolder)

    os.makedirs(full_image_path, exist_ok=True)
    
    pdf_dest_path = os.path.join(full_image_path, f"{safe_name}.pdf")
    shutil.copy(pdf_path, pdf_dest_path)
    

    #model, tokenizer = load(BASE_MODEL)
    sections = parse_pdf_sections_robust(pdf_path, image_subfolder, full_image_path)
    
    intro_text = sections[0][1] + "\n" + (sections[1][1] if len(sections)>1 else "")

    concepts = extract_concepts(intro_text)
    paper_title = extract_paper_title(paper_full_text, paper_name)
    
    if paper_title is None:
        print("Failed to extract paper title")
        return
    
    with open(output_file, "w") as f:
        f.write(f"# {paper_title}\n")
        f.write(f"**Connected Concepts:** {concepts}\n\n")
        f.write(f"[[assets/{safe_name}/{safe_name}.pdf|{safe_name}]] (PDF Source)\n\n---\n")
    
    style_content = {}
    header_visuals = {}

    # --- VISUAL EXTRACTION LOOP ---
    # Do this ONCE before loading models to save compute
    print("🖼️ Extracting Visuals...")
    for header, content in sections:
        visuals = extract_visuals_only(content)
        if visuals:
            header_visuals[header] = visuals

    for style, config in STYLE_CONFIG.items():
        print(f"Processing {style}")
        try:
            model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTERS[style])

            # CRITICAL: Set EOS token for Llama 3
            if "<|eot_id|>" in tokenizer.get_vocab():
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

            for header,content in sections:
                if any(x in header.lower() for x in ["reference", "citation", "acknowledg","bibliography"]):
                    print(f"Skipping {header}")
                    continue
                print(f"Working on {header}")
                prompt_text = f"{config['prompt']}\n\nText:\n{content}"
                messages = [{"role": "user", "content": prompt_text}]
                
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                response = generate(model, tokenizer, prompt=prompt, max_tokens=1000)

                cleaned_response = response.replace("\n", "\n> ")
                if header not in style_content:
                    style_content[header] = {}
                style_content[header][style] = cleaned_response
        except Exception as e:  
            print(f"Failed {style}: {e}")
            continue
    
    for header in style_content:
        print(f"Writing {header}")
        if any(x in header.lower() for x in ["reference", "citation", "acknowledg","bibliography"]):
            continue

        with open(output_file, "a") as f:
            f.write(f"\n## {header}\n")
            
            if header in header_visuals:
                f.write("> [!ABSTRACT]- Visuals (Click to expand)\n")
                for vis in header_visuals[header]:
                    indented_vis = vis.replace("\n", "\n> ")
                    f.write(f"> {indented_vis}\n> \n")
                f.write("---\n") # Separator line

            for style, config in STYLE_CONFIG.items():
                try:
                    f.write(f"{config['callout']}\n")
                    f.write(f"> {style_content[header][style]}\n\n")
                    
                except Exception as e:
                    print(f"Failed {style}: {e}")
                    continue
            
            
if __name__ == "__main__":
    main()
    


    