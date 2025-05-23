<h1 align="center">The Structural Safety Generalization Problem</h1>
<h3 align="center">Findings of ACL 2025</h3>
<p align='center' style="text-align:center;font-size:1em;">
Julius Broomfield*¹, Tom Gibbs*², Ethan Kosak-Hine*², George Ingebretsen*³, Tia Nasir, Jason Zhang⁴, Reihaneh Iranmanesh³, Sara Pieri⁵, Reihaneh Rabbany²'⁶, Kellin Pelrine²'⁶
</p>
<p align='center' style="text-align:center;font-size:0.8em;">
¹Georgia Tech, ²Mila, ³UC Berkeley, ⁴Stanford, ⁵MBZUAI, ⁶McGill
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2504.09712-b31b1b.svg?style=flat)](https://arxiv.org/abs/2504.09712)

## 📖 Abstract

LLM jailbreaks remain a major safety challenge. We tackle this by focusing on a specific failure mode: safety mechanisms don't generalize well across semantically equivalent inputs. Our approach identifies attacks that are explainable and transferable across both models and goals.

Through red-teaming, we discovered new vulnerabilities using multi-turn conversations, multiple images, and translation-based attacks. Different input structures lead to different safety outcomes. Based on these findings, we developed a Structure Rewriting Guardrail that converts inputs into formats better suited for safety assessment. This defense significantly improves rejection of harmful inputs without over-blocking legitimate ones.

By targeting this intermediate challenge—more achievable than universal defenses but crucial for long-term safety—we establish an important milestone for AI safety research.

## ⚙️ Installation

This project uses Poetry for dependency management and packaging.

### 1. Clone the repository.
   
   ```bash
    git clone https://github.com/juliusbroomfield/the-SSG-problem.git
    cd the-SSG-problem
   ```

### 2. Install the required packages.
   
   ```bash
   poetry install
   ```
### 3. Configure API Keys
Create a .env file in the project root and add your LLM provider API keys and other neccessary parameters. 

This project uses [LiteLLM](https://docs.litellm.ai/docs/). A list of supported models and providers can be found at https://docs.litellm.ai/docs/providers.

For example, if using Azure OpenAI:

 ```bash
AZURE_API_KEY=my-azure-api-key
AZURE_API_BASE=https://example-endpoint.openai.azure.com
AZURE_API_VERSION=2023-05-15
```

## 🚀 Usage

The primary entry point is the `inferencing.py` script.

**Basic Usage:**
```bash
poetry run python inferencing.py \
  --data_file datasets/harmful/multiturn/dataset.json \
  --output_dir outputs/experiment1 \
  --model_name "gpt-4o-mini" \
  --runs 3
```

### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--data_file` | Path to the JSON dataset |
| `--output_dir` | Directory for saving results |
| `--model_name` | Model to evaluate |
| `--api_key` | API key |
| `--use_guardrail` | Enable guardrail wrapper for refusal detection |
| `--strongreject` | Run StrongReject safety evaluation |
| `--strongreject_model` | Model for safety evaluation |
| `--strongreject_prompt` | Custom safety evaluation prompt |
| `--methods` | Specific method types to process |
| `--runs` | Number of response generations per prompt |

## 📊 Datasets

### Multi-Turn Safety Dataset
- **Size**: 24,816 examples
- **Base**: 4,136 harmful instructions from AdvBench with conversational priming
- **Languages**: English, Welsh, Tamil
- **Variations**: Single-turn vs. multi-turn structures
- **Benign Subset**: 2,400 examples (partially benign + completely benign)

### Multi-Modal Safety Dataset
- **Size**: 6,500 harmful examples + 90 benign examples  
- **Categories**: Harmful Content, Malicious Activities, Dangerous Substances, Misinformation, Explicit Content
- **Languages**: English, Welsh, Tamil
- **Structural Variations**: 
  - Plain text vs. unperturbed & perturbed variants
  - Composite & decomposite variants
  - Color substitution cipher
  - Image-based prompt decomposition

### Dataset Structure
```
datasets/
├── harmful/
│   ├── multiturn/
│   │   └── dataset.json
│   └── multimodal/
│       ├── dataset.json
│       └── images/
└── benign/
    ├── multiturn/
    └── multimodal/
```

Each dataset entry contains:
- **ID**: Unique identifier
- **Type**: Method/variation type
- **Prompt**: Input text (may reference images)
- **Full Phrase**: Original English instruction
- **Category/Subcategory**: Classification labels
- **Images**: Paths to associated image files

### 📝 Output Format

Results are saved in both JSON and CSV formats:
```
{model_name}-{timestamp}.json
{model_name}-{timestamp}.csv
```

## 🤝 Citation

If you find our work helpful, please cite our paper:

```bibtex
@misc{broomfield2025structuralsafetygeneralizationproblem,
      title={The Structural Safety Generalization Problem}, 
      author={Julius Broomfield and Tom Gibbs and Ethan Kosak-Hine and George Ingebretsen and Tia Nasir and Jason Zhang and Reihaneh Iranmanesh and Sara Pieri and Reihaneh Rabbany and Kellin Pelrine},
      year={2025},
      eprint={2504.09712},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2504.09712}, 
}
```
