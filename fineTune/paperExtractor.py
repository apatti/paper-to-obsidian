

import arxiv
import pymupdf4llm
import json
import os

def fetch_papers(category: str, max_papers: int = 10, save_dir: str = "papers") -> list:

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image_dir = os.path.join(save_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    print(f"Fetching up to {max_papers} papers from category: {category}")

    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    client = arxiv.Client()
    
    papers = []
    for result in client.results(search):

        safe_title = "".join(c for c in result.title if c.isalnum() or c.isspace()).rstrip()
        pdf_path = os.path.join(save_dir, f"{safe_title}.pdf")
        
        # 1. Download the PDF
        print(f"Downloading {result.title}...")
        result.download_pdf(dirpath=save_dir, filename=f"{safe_title}.pdf")

        # 2. Extract Text + Diagrams
        # 'write_images=True' extracts both photos and vector charts
        # 'image_path' tells it where to save them
        md_text = pymupdf4llm.to_markdown(
            pdf_path,
            image_path=image_dir,
            write_images=True,
            image_format="png"
        )

        papers.append({
            "title": result.title,
            "full_text": md_text,
            "pdf_path": pdf_path
        })
    
    return papers

if __name__ == "__main__":
    papers = []
    categories = ["cs.LG", "cs.AI", "cs.DS", "cs.GT", "cs.PF", "cs.SE", "cs.OH"]
    for category in categories:
        papers.extend(fetch_papers(category, max_papers=7))
    print(f"Downloaded {len(papers)} papers")
    for paper in papers:
        print(f"- {paper['title']}")

    # Save to JSON for later use
    with open("papers.json", "w") as f:
        json.dump(papers, f, indent=2)


    print(f"Success! Saved {len(papers)} papers to papers.json")