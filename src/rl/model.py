import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharIdiomEmbedding(nn.Module):
    """Character-level idiom embedding: each idiom = 4 chars, embedded then projected."""

    def __init__(self, n_chars, char_dim=64, idiom_dim=256):
        super().__init__()
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(char_dim * 4, idiom_dim),
            nn.LayerNorm(idiom_dim),
        )
        self.register_buffer('idiom_chars', torch.zeros(1, 4, dtype=torch.long))

    def set_idiom_chars(self, idiom_chars_np):
        self.idiom_chars = torch.from_numpy(idiom_chars_np).long()

    def forward(self, idiom_ids):
        # idiom_id=0 is used for both padding and the real idiom at index 0.
        # Padded positions are masked out by history_mask / candidate_mask
        # in downstream attention and logit computation, so this is safe.
        safe_ids = idiom_ids.clamp(0, self.idiom_chars.size(0) - 1)
        chars = self.idiom_chars[safe_ids]
        char_embs = self.char_emb(chars)
        flat = char_embs.flatten(start_dim=-2)
        return self.proj(flat)


class CrossAttentionEncoder(nn.Module):
    """Cross-attention history encoder. Current node queries history set."""

    def __init__(self, idiom_dim=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.role_emb = nn.Embedding(2, idiom_dim)
        self.player_emb = nn.Embedding(2, idiom_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    idiom_dim, n_heads, dropout=dropout, batch_first=True
                ),
                'norm1': nn.LayerNorm(idiom_dim),
                'ffn': nn.Sequential(
                    nn.Linear(idiom_dim, idiom_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(idiom_dim * 2, idiom_dim),
                    nn.Dropout(dropout),
                ),
                'norm2': nn.LayerNorm(idiom_dim),
            }))
        self.empty_token = nn.Parameter(torch.randn(1, 1, idiom_dim) * 0.02)

    def forward(self, u_emb, hist_emb, hist_mask, player_ids):
        """player_ids: (B,) tensor with values 0 or 1 indicating current player."""
        batch_size = u_emb.size(0)

        empty = self.empty_token.expand(batch_size, -1, -1)
        hist_emb = torch.cat([empty, hist_emb], dim=1)
        empty_mask = torch.ones(batch_size, 1, dtype=torch.bool,
                                device=hist_mask.device)
        hist_mask = torch.cat([empty_mask, hist_mask], dim=1)

        u_emb = u_emb + self.role_emb.weight[0] + self.player_emb(player_ids)
        hist_emb = hist_emb + self.role_emb.weight[1]

        query = u_emb.unsqueeze(1)
        kv_padding_mask = ~hist_mask

        for layer in self.layers:
            attended, _ = layer['cross_attn'](
                query, hist_emb, hist_emb,
                key_padding_mask=kv_padding_mask
            )
            query = layer['norm1'](query + attended)
            query = layer['norm2'](query + layer['ffn'](query))

        return query.squeeze(1)


class PolicyValueNet(nn.Module):
    """Full policy-value network for idiom solitaire."""

    def __init__(self, n_idioms, n_chars,
                 idiom_dim=256, n_heads=4, n_layers=2,
                 encoder_type='cross_attention',
                 embedding_type='char'):
        super().__init__()

        if embedding_type == 'char':
            self.idiom_emb = CharIdiomEmbedding(n_chars, char_dim=64,
                                                idiom_dim=idiom_dim)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        if encoder_type == 'cross_attention':
            self.encoder = CrossAttentionEncoder(idiom_dim, n_heads, n_layers)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.value_head = nn.Sequential(
            nn.Linear(idiom_dim, idiom_dim // 2),
            nn.GELU(),
            nn.Linear(idiom_dim // 2, 1),
            nn.Tanh(),
        )

        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, u_ids, history_ids, history_mask,
                candidate_ids, candidate_mask, player_ids):
        u_emb = self.idiom_emb(u_ids)
        hist_emb = self.idiom_emb(history_ids)
        cand_emb = self.idiom_emb(candidate_ids)

        state = self.encoder(u_emb, hist_emb, history_mask, player_ids)
        value = self.value_head(state).squeeze(-1)

        temperature = self.log_temperature.exp().clamp(0.1, 10.0)
        logits = torch.einsum('bd,bcd->bc', state, cand_emb) / temperature
        logits = logits.masked_fill(~candidate_mask, float('-inf'))

        return logits, value

    @torch.no_grad()
    def get_action(self, u_ids, history_ids, history_mask,
                   candidate_ids, candidate_mask, player_ids,
                   deterministic=False):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, value = self.forward(
                u_ids, history_ids, history_mask,
                candidate_ids, candidate_mask, player_ids
            )
        logits = logits.float()
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action_idx = probs.argmax(dim=-1)
        else:
            action_idx = torch.multinomial(probs, 1).squeeze(-1)

        log_probs_all = F.log_softmax(logits, dim=-1)
        log_prob = log_probs_all.gather(1, action_idx.unsqueeze(1)).squeeze(1)

        return action_idx, log_prob, value


def test_model():
    """Verify all model variants forward pass correctly."""
    configs = [
        ('cross_attention', 'char'),
    ]

    for enc_type, emb_type in configs:
        model = PolicyValueNet(
            n_idioms=100, n_chars=50, idiom_dim=64,
            n_heads=2, n_layers=1,
            encoder_type=enc_type, embedding_type=emb_type
        )
        model.idiom_emb.set_idiom_chars(np.random.randint(1, 50, (100, 4)))

        B = 4
        u = torch.randint(0, 100, (B,))
        h = torch.randint(0, 100, (B, 10))
        h_mask = torch.ones(B, 10, dtype=torch.bool)
        h_mask[:, 5:] = False
        c = torch.randint(0, 100, (B, 20))
        c_mask = torch.ones(B, 20, dtype=torch.bool)
        c_mask[:, 15:] = False
        p = torch.randint(0, 2, (B,))

        logits, values = model(u, h, h_mask, c, c_mask, p)
        assert logits.shape == (B, 20), f"{enc_type}/{emb_type}: logits shape error"
        assert values.shape == (B,), f"{enc_type}/{emb_type}: values shape error"
        assert not torch.isnan(logits).any(), f"{enc_type}/{emb_type}: logits has NaN"
        assert not torch.isnan(values).any(), f"{enc_type}/{emb_type}: values has NaN"
        assert (logits[:, 15:] == float('-inf')).all(), \
            f"{enc_type}/{emb_type}: mask not correctly applied"

        # Test empty history
        h_empty = torch.zeros(B, 10, dtype=torch.long)
        h_mask_empty = torch.zeros(B, 10, dtype=torch.bool)
        logits2, values2 = model(u, h_empty, h_mask_empty, c, c_mask, p)
        assert not torch.isnan(logits2).any(), \
            f"{enc_type}/{emb_type}: empty history caused NaN"

        print(f"  {enc_type}/{emb_type}: OK")

    print("All model variant tests PASSED")


if __name__ == '__main__':
    test_model()
