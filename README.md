# Qwen3 Small: Story Generator & Sentiment Analyzer

A custom 200M parameter transformer model trained from scratch for creative text generation and sentiment analysis.

[![Demo](https://img.shields.io/badge/🤗-Hugging%20Face%20Demo-blue)](https://huggingface.co/spaces/YOUR_USERNAME/qwen3-small-demo)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

This project implements a custom Qwen3-inspired transformer architecture with:
- **Text Generation**: Creative storytelling with temperature-controlled sampling
- **Sentiment Analysis**: Binary classification (positive/negative) using frozen embeddings
- **Custom Training**: Trained on 100K TinyStories from scratch

**Key Metrics:**
- 200M trainable parameters
- 82% sentiment classification accuracy
- 2.45 validation loss on story generation

## 🚀 Live Demo

Try the model here: [Hugging Face Space](https://huggingface.co/spaces/YOUR_USERNAME/qwen3-small-demo)

## 📊 Features

### Text Generation
- Creative story completion from prompts
- Temperature control (0.1 - 2.0) for creativity vs. coherence
- Trained on TinyStories dataset (child-friendly narratives)

### Sentiment Analysis
- Binary sentiment classification (positive/negative)
- Uses frozen language model embeddings
- Lightweight 3-layer MLP classifier

## 🏗️ Architecture

- **Model**: Custom Qwen3 with Grouped Query Attention
- **Embedding Dim**: 896
- **Layers**: 16 transformer blocks
- **Attention Heads**: 14 (7 KV groups)
- **Context Length**: 1024 tokens
- **Vocab Size**: 8000 (SentencePiece tokenizer)
- **Position Encoding**: RoPE (Rotary Position Embedding)

## 📁 Project Structure

```
├── Trevor_Kunz_Midterm.ipynb   # Training notebook
├── app.py                       # Gradio demo application
├── requirements.txt             # Python dependencies
├── qwen3_tokenizer.model        # SentencePiece tokenizer
├── qwen3_small_trained.pth      # Trained model weights (not in repo)
├── sentiment_classifier.pth     # Classifier weights (not in repo)
└── README.md                    # This file
```

**Note**: Model checkpoint files (`.pth`) are too large for GitHub. Download from [Hugging Face Space](https://huggingface.co/spaces/YOUR_USERNAME/qwen3-small-demo/tree/main).

## 🛠️ Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qwen3-small.git
cd qwen3-small

# Install dependencies
pip install -r requirements.txt

# Download model files from Hugging Face Space
# (Or train from scratch using the notebook)

# Run the Gradio app
python app.py
```

### Training from Scratch

Open `Trevor_Kunz_Midterm.ipynb` in Jupyter/Colab and run all cells:

1. **Tokenizer Training**: Creates custom SentencePiece tokenizer
2. **Model Training**: Trains language model on TinyStories
3. **Classifier Training**: Trains sentiment classifier on frozen embeddings

**Requirements:**
- GPU recommended (A100 for ~5 hours total training)
- ~10GB GPU memory
- ~15GB disk space for datasets

## 📈 Training Details

### Language Model
- **Dataset**: TinyStories (60K train / 20K val / 20K test)
- **Optimizer**: AdamW (lr=3e-4, weight decay=0.1)
- **Batch Size**: 16
- **Max Sequence Length**: 256 tokens
- **Early Stopping**: Patience of 5 validation checks

### Sentiment Classifier
- **Dataset**: Emotions dataset (~5K samples)
- **Architecture**: 896 → 256 → 128 → 2
- **Training**: 20 epochs with frozen LM
- **Test Accuracy**: 82.5%

## 💻 Usage Examples

### Text Generation

```python
from app import generate_text

text = generate_text(
    prompt="Once upon a time",
    max_tokens=50,
    temperature=0.8
)
print(text)
```

### Sentiment Analysis

```python
from app import analyze_sentiment

sentiment = analyze_sentiment("I love this beautiful day!")
print(sentiment)  # "Positive (Confidence: 95.3%)"
```

## 🎓 Educational Context

This project was created as a Generative AI midterm assignment, demonstrating:
- Custom transformer implementation from scratch
- Training on domain-specific datasets
- Multi-task learning (generation + classification)
- Transfer learning with frozen embeddings
- Production deployment with Gradio

## 📚 Reference

The Qwen3 architecture implementation is inspired by:
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb) by Sebastian Raschka
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609) by Alibaba Cloud

## 🔗 Links

- **Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/YOUR_USERNAME/qwen3-small-demo)
- **Portfolio**: [Your Portfolio](https://yourportfolio.vercel.app)
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## ⚖️ License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- TinyStories dataset by Eldan & Li (2023)
- Qwen architecture by Alibaba Cloud
- Built with PyTorch, Gradio, and Hugging Face
