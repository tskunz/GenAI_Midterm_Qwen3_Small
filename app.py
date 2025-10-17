"""
Qwen3 Small - Text Generation & Sentiment Analysis
A custom-trained transformer model for creative storytelling and sentiment classification.
Trained on TinyStories dataset with ~200M parameters.
"""

import gradio as gr
import torch
import torch.nn as nn
import sentencepiece as spm
import os

# ============================================================================
# MODEL ARCHITECTURE (Same as notebook)
# ============================================================================

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.last_hidden_state = None
        self._hook_enabled = False

    def enable_embedding_hook(self):
        self._hook_enabled = True

    def disable_embedding_hook(self):
        self._hook_enabled = False

    def get_last_hidden_state(self):
        return self.last_hidden_state

    def forward(self, in_idx, return_embeddings=False):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        if self._hook_enabled or return_embeddings:
            self.last_hidden_state = x
        if return_embeddings:
            return x
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


class SentimentClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)


class SimpleTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp.vocab_size()
        self.eos_token_id = self.sp.eos_id()
        self.pad_token_id = self.sp.pad_id()
        self.bos_token_id = self.sp.bos_id()
        self.unk_token_id = self.sp.unk_id()

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)


# ============================================================================
# MODEL LOADING
# ============================================================================

print("Loading models...")

# Configuration
QWEN3_CONFIG = {
    "vocab_size": 8000,  # Will be updated from tokenizer
    "context_length": 1024,
    "emb_dim": 896,
    "n_heads": 14,
    "n_layers": 16,
    "hidden_dim": 3584,
    "head_dim": 64,
    "qk_norm": True,
    "n_kv_groups": 7,
    "rope_base": 10_000.0,
    "dtype": torch.bfloat16,
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = SimpleTokenizer("qwen3_tokenizer.model")
QWEN3_CONFIG["vocab_size"] = tokenizer.vocab_size

# Load language model
model = Qwen3Model(QWEN3_CONFIG)
checkpoint = torch.load("qwen3_small_trained.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load sentiment classifier
classifier = SentimentClassifier(embedding_dim=QWEN3_CONFIG['emb_dim'], num_classes=2)
classifier_checkpoint = torch.load("sentiment_classifier.pth", map_location=device)
classifier.load_state_dict(classifier_checkpoint['classifier_state_dict'])
classifier.to(device)
classifier.eval()

print("Models loaded successfully!")

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def generate_text(prompt, max_tokens=50, temperature=0.8):
    """Generate text from a prompt"""
    model.eval()

    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids, device=device).unsqueeze(0)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(token_ids)[:, -1, :]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token.item())
            token_ids = torch.cat([token_ids, next_token], dim=1)

    full_text = tokenizer.decode(tokenizer.encode(prompt) + generated_tokens)
    return full_text


def analyze_sentiment(text):
    """Analyze sentiment of text"""
    model.eval()
    classifier.eval()

    # Extract embedding
    tokens = tokenizer.encode(text)
    max_length = 128

    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    if len(tokens) < max_length:
        tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))

    input_ids = torch.tensor([tokens], device=device)

    with torch.no_grad():
        # Get embeddings
        hidden_states = model(input_ids, return_embeddings=True)

        # Mean pooling
        mask = (input_ids != tokenizer.pad_token_id).float().unsqueeze(-1)
        masked_hidden = hidden_states * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1)
        embedding = (summed / counts)

        # Classify
        outputs = classifier(embedding)
        probs = torch.softmax(outputs, dim=-1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()

    sentiment = "Positive" if prediction == 1 else "Negative"
    return f"{sentiment} (Confidence: {confidence:.2%})"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
}
.gr-button {
    color: white;
    border-color: #9D2235;
    background: #9D2235;
}
.gr-button:hover {
    border-color: #8B1E2F;
    background: #8B1E2F;
}
"""

# Text generation interface
with gr.Blocks(css=custom_css, title="Qwen3 Small - Story Generator & Sentiment Analyzer") as demo:
    gr.Markdown("""
    # 🤖 Qwen3 Small: Story Generator & Sentiment Analyzer

    A custom-trained transformer model with ~200M parameters, trained on the TinyStories dataset.
    This demo showcases two capabilities:
    1. **Text Generation**: Creative storytelling with temperature-controlled sampling
    2. **Sentiment Analysis**: Binary classification (positive/negative) using frozen embeddings

    ---
    """)

    with gr.Tab("📝 Text Generation"):
        gr.Markdown("### Generate creative stories from a prompt")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Story Prompt",
                    placeholder="Once upon a time...",
                    lines=3
                )
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature (higher = more creative)",
                )
                max_tokens_slider = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=10,
                    label="Max Tokens to Generate",
                )
                generate_btn = gr.Button("Generate Story", variant="primary")

            with gr.Column():
                text_output = gr.Textbox(
                    label="Generated Text",
                    lines=10,
                    interactive=False
                )

        gr.Examples(
            examples=[
                ["Once upon a time", 0.8, 50],
                ["The princess was lost in the forest", 0.7, 40],
                ["On a sunny day, the children went to", 0.9, 60],
            ],
            inputs=[text_input, temperature_slider, max_tokens_slider],
        )

        generate_btn.click(
            fn=generate_text,
            inputs=[text_input, max_tokens_slider, temperature_slider],
            outputs=text_output
        )

    with gr.Tab("💭 Sentiment Analysis"):
        gr.Markdown("### Analyze the sentiment of any text")

        with gr.Row():
            with gr.Column():
                sentiment_input = gr.Textbox(
                    label="Text to Analyze",
                    placeholder="I love this beautiful day!",
                    lines=5
                )
                analyze_btn = gr.Button("Analyze Sentiment", variant="primary")

            with gr.Column():
                sentiment_output = gr.Textbox(
                    label="Sentiment Result",
                    lines=2,
                    interactive=False
                )

        gr.Examples(
            examples=[
                ["I love this beautiful day!"],
                ["This is terrible and I hate it."],
                ["The weather is okay, I guess."],
            ],
            inputs=sentiment_input,
        )

        analyze_btn.click(
            fn=analyze_sentiment,
            inputs=sentiment_input,
            outputs=sentiment_output
        )

    gr.Markdown("""
    ---
    ### 📊 Model Details
    - **Architecture**: Custom Qwen3 with Grouped Query Attention
    - **Parameters**: ~200M (896 embedding dim, 16 layers, 14 attention heads)
    - **Training**: TinyStories dataset (100K stories)
    - **Features**: RoPE positional encoding, RMSNorm, SiLU activations

    Built by [Your Name] | [GitHub](https://github.com/yourusername) | [Portfolio](https://yourportfolio.com)
    """)

if __name__ == "__main__":
    demo.launch()
