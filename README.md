# Paper to Obsidian

Turn dense research papers into beautiful, interconnected Obsidian notes locally on your Mac.

Paper to Obsidian is an AI-powered tool that transforms academic research papers into richly formatted Obsidian markdown notes. It uses a local Mixture-of-Experts pipeline to analyze every section of a paper through three distinct lenses, making complex research accessible at multiple levels of comprehension—all running 100% offline on Apple Silicon.

## Features

- **Multi-Perspective Analysis**: Every section is explained through three AI personas:
  - **ELI5**: Simple, analogy-based explanations for beginners
  - **Intuitive**: First-principles engineering explanations (Feynman-style)
  - **Executive**: Business/utility-focused summaries for quick understanding

- **Vision AI Captioning**: Automatically describes charts, diagrams, and figures using a vision-language model

- **Obsidian Integration**: Generates markdown with `[[wiki-links]]` for knowledge graph connectivity

- **Efficient Local Inference**: Runs entirely on Apple Silicon using MLX with quantized models

## Requirements

- Python 3.12+
- macOS with Apple Silicon (M1/M2/M3/M4)
- ~8GB RAM for model inference

## Installation

```bash
# Clone the repository
git clone https://github.com/apatti/paper-to-obsidian.git
cd paper-to-obsidian

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```bash
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault
```

## Usage

```bash
python obsidian_paper.py /path/to/paper.pdf
```

This will:
1. Parse the PDF and extract all images
2. Caption figures using the vision model
3. Identify key concepts and technical terms
4. Generate three explanations per section (ELI5, Intuitive, Executive)
5. Output a formatted markdown file to your Obsidian vault

## Output Example

Your Obsidian note will look like this:

```markdown
# Paper Title

**Connected Concepts**: [[Transformer]] | [[Attention Mechanism]] | [[BERT]]

---

## Abstract

![Figure 1](assets/paper_name/image_0.png)
> [!INFO] AI Vision
> *A diagram showing the transformer architecture with encoder and decoder blocks...*

> [!TIP] ELI5 Analogy
> Imagine you're reading a book but you can look at any page instantly...

> [!EXAMPLE] Engineering Logic
> The key mechanism here is self-attention, which computes weighted sums...

> [!SUMMARY] Executive Verdict
> This approach reduces training time by 10x while maintaining accuracy...
```

## Architecture

```
graph TD
    A[PDF Input] --> B[Layout Parsing\npymupdf4llm];
    B --> C[Vision Pass\nQwen2-VL-2B];
    C --> D{Matrix Analysis};
    D -->|Load| E[Executive Adapter];
    D -->|Load| F[Intuitive Adapter];
    D -->|Load| G[ELI5 Adapter];
    E & F & G --> H[Obsidian Vault];
    H --> I[Graph Connections];
```

### Models Used

| Component | Model | Function | Size |
|-----------|-------|------|------|
| Base LLM | Llama 3.2 3B Instruct (4-bit) | The reasoning engine| ~2GB |
| Vision | Qwen2-VL-2B Instruct (4-bit) | Image captioning | ~1.5GB |
| Adapters | LoRA fine-tuned | The "Personas" | ~8MB each |

## Fine-Tuning

See [fineTune/README.md](fineTune/README.md) for instructions on creating custom adapters.

## Project Structure

```
paper-to-obsidian/
|-- obsidian_paper.py       # Main entry point
|-- utils/
|   |-- vision.py           # Vision model captioning
|-- adapters/               # Fine-tuned LoRA adapters
|   |-- eli5_final/
|   |-- executive_final/
|   |-- intuitive_final/
|-- fineTune/               # Training pipeline
|-- .env                    # Configuration (create this)
|-- pyproject.toml          # Dependencies
|-- LICENSE                 # MIT License
```

## Dependencies

Key dependencies include:
- `mlx-lm` / `mlx-vlm`: Apple Silicon ML inference
- `pymupdf4llm`: PDF to Markdown conversion
- `python-dotenv`: Environment configuration

See `pyproject.toml` for the complete list.

## License

MIT License - see [LICENSE](LICENSE) for details.
