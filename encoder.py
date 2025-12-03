import torch
import torch.nn as nn

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=256):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention + residual + normalization
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward + residual + normalization
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


# Test block
if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    d_model = 128

    encoder = SimpleTransformerEncoder(d_model=d_model)

    dummy_input = torch.randn(batch_size, seq_len, d_model)

    output = encoder(dummy_input)

    print("Output shape:", output.shape)
