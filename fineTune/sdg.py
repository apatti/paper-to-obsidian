from dotenv import load_dotenv
from google import genai
import re

import os
from typing import List, Dict
import json
import time
from tqdm import tqdm

def extract_sections(markdown_text: str, skip_references: bool = True) -> List[Dict]:
    """
    Extract sections from markdown text.

    Args:
        markdown_text: The markdown content to parse
        skip_references: If True, skip sections with "reference" in title

    Returns:
        List of dictionaries with 'level', 'title', and 'content'.
    """
    sections = []
    
    major_section_pattern = r'\*\*(\d+)\*\*\s+\*\*([^*]+)\*\*'
    
    # Find all major section headers
    matches = list(re.finditer(major_section_pattern, markdown_text))
    
    for i, match in enumerate(matches):
        section_num = match.group(1)
        section_title = match.group(2).strip()
        
        # Get content start
        start_pos = match.end()
        
        # Get content end (next major section or end of text)
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(markdown_text)
        
        # Extract and clean content
        content = markdown_text[start_pos:end_pos].strip()
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        sections.append({
            'section_number': section_num,
            'title': section_title,
            'content': content
        })

    # Apply filters
    filtered_sections = []
    for i, section in enumerate(sections):

        # Skip references section if skip_references is True
        if skip_references and 'reference' in section['title'].lower():
            continue

        filtered_sections.append(section)

    return filtered_sections


load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

panel_prompt = """
        You are a panel of three experts analyzing a technical research paper section.
        Input Text:  
        {section_content}

        Your goal is to provide three distinct outputs in a STRICT JSON format:

        1. "ELI5" (The Kindergarten Teacher): You are an expert kindergarten teacher and master of analogies. Your goal is to explain complex academic text to a bright 5-year-old.
            RULES:
            -  Use specific, physical analogies (e.g., "like a postman delivering mail" instead of "data transmission").
            -  Forbidden words: "optimize", "leverage", "paradigm", "stochastic", "neural".
            -  Focus ONLY on the 'Big Idea', ignoring the math details.

        2. "Intuitive" (The First-Principles Engineer): You are a First Principles engineer (like Richard Feynman). Your goal is to strip away the jargon and explain *why* this works mathematically, but intuitively.
            RULES:
            -  Don't just simplify; reveal the mechanism. 
            -  If you see an equation (e.g., softmax), explain the *behavior* (e.g., "it forces one winner to take all the votes").
            -  Connect new concepts to standard programming concepts (hash maps, loops, pointers).

        3. "Executive" (The VC Analyst): You are a ruthless Venture Capitalist analyst.  Your goal is to determine the *value* and *utility* of this section.
            RULES:
            -  BLUF (Bottom Line Up Front).
            -  Highlight: Costs, Hardware Requirements, and Utility.
            -  Ignore theoretical novelties unless they improve speed or cost.
        
        OUTPUT SCHEMA (JSON):
        {{
            "eli5": {{
                "analogy": "...",
                "explanation": "..."
            }},
            "intuitive": {{
                "mechanism": "...",
                "explanation": "..."
            }},
            "executive": {{
                "verdict": "...",
                "summary": "..."
            }}
        }}
        """

# Define the JSON schema for structured output
response_schema = {
    "type": "object",
    "properties": {
        "eli5": {
            "type": "object",
            "properties": {
                "analogy": {"type": "string"},
                "explanation": {"type": "string"}
            },
            "required": ["analogy", "explanation"]
        },
        "intuitive": {
            "type": "object",
            "properties": {
                "mechanism": {"type": "string"},
                "explanation": {"type": "string"}
            },
            "required": ["mechanism", "explanation"]
        },
        "executive": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string"},
                "summary": {"type": "string"}
            },
            "required": ["verdict", "summary"]
        }
    },
    "required": ["eli5", "intuitive", "executive"]
}

def process_section_text(sectionText: str) -> dict:
    try:
        prompt = panel_prompt.format(section_content=sectionText)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        )

        #print(f"Response: {response.text}")
        return json.loads(response.text)
    except Exception as e:
        print(f"Error processing section: {e}")
        return None

def save_jsonl(filename, data):
    with open(filename, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def generate_sdg(papers: List[Dict[str, str]]) -> Dict[str, List[Dict]]:
    eli5_data = []
    intuitive_data = []
    executive_data = []
    for i, paper in enumerate(tqdm(papers, desc="Processing papers")):
        full_text = paper.get('full_text', '')
        sections = extract_sections(full_text)
        
        for section in tqdm(sections, desc=f"Paper {i+1}/{len(papers)}", leave=False):
            if len(section['content']) < 100:
                print(f"Skipping short section: {section['title']}")
                continue
            # Process each section here
            #print(f"Processing section: {section['title']}")

            data = process_section_text(section['content'])

            if data:
                eli5_entry = {
                    "messages": [
                        {"role": "user", "content": f"Explain this like I'm 5:\n{section['content']}"},
                        {"role": "assistant", "content": f"**Analogy:** {data['eli5']['analogy']}\n\n**Explanation:** {data['eli5']['explanation']}"}
                    ]
                }
                eli5_data.append(eli5_entry)

                int_entry = {
                    "messages": [
                        {"role": "user", "content": f"Explain the intuition behind this:\n{section['content']}"},
                        {"role": "assistant", "content": f"**Mechanism:** {data['intuitive']['mechanism']}\n\n**Explanation:** {data['intuitive']['explanation']}"}
                    ]
                }
                intuitive_data.append(int_entry)

                executive_entry = {
                    "messages": [
                        {"role": "user", "content": f"Give me an executive summary:\n{section['content']}"},
                        {"role": "assistant", "content": f"**Verdict:** {data['executive']['verdict']}\n\n**Summary:** {data['executive']['summary']}"}
                    ]
                }
                executive_data.append(executive_entry)

                time.sleep(1)  # Be respectful to the API
    
        # Save the data to separate JSONL files
        os.makedirs('data', exist_ok=True)
        save_jsonl('data/eli5.jsonl', eli5_data)
        save_jsonl('data/intuitive.jsonl', intuitive_data)
        save_jsonl('data/executive.jsonl', executive_data)
    
    return {
        "eli5": eli5_data,
        "intuitive": intuitive_data,
        "executive": executive_data
    }

if __name__ == "__main__":
    #read from papers.json
    with open('papers.json', 'r') as f:
        papers = json.load(f)
        result = generate_sdg(papers)
        print("SDG generation complete!")
        print(f"Generated {len(result['eli5'])} ELI5 entries")
        print(f"Generated {len(result['intuitive'])} intuitive entries")
        print(f"Generated {len(result['executive'])} executive entries")