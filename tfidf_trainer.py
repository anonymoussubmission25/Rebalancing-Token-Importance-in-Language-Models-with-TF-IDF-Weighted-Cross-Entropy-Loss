import torch
from transformers import Trainer


class TfidfLossTrainer(Trainer):
    """
    Extends HuggingFace Trainer with a TF-IDF weighted cross-entropy objective.

    Token-level loss weights are computed using local TF and smoothed IDF estimated
    over a rolling buffer of the K most recent mini-batches (N = B * K sequences).
    Weights are normalized to unit mean so standard learning rates remain applicable.
    """

    def __init__(self, *args, K=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = K
        self._buf = None        # (N_max, L) ring buffer of token ids, on device
        self._buf_ptr = 0       # next write position (mod BUF_SIZE)
        self._buf_size = None   # B * K, set on first batch

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()           # (B, L)
        mask = (shift_labels != -100)                       # (B, L)
        num_valid = mask.sum()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        B, L = shift_labels.shape
        device = shift_labels.device
        vocab_size = model.config.vocab_size

        # --- Initialise ring buffer on first call ---
        if self._buf is None:
            self._buf_size = B * self.K
            self._buf = torch.zeros(self._buf_size, L, dtype=torch.long, device=device)

        # --- Update ring buffer (one row per sequence) ---
        for i in range(B):
            self._buf[self._buf_ptr % self._buf_size] = shift_labels[i]
            self._buf_ptr += 1
        N = min(self._buf_ptr, self._buf_size)
        active_buf = self._buf[:N]                          # (N, L)

        # --- TF: occurrences of each token within its own sequence ---
        # Uses bincount per row; result indexed back via shift_labels.
        tf_weights = torch.zeros(B, L, device=device, dtype=torch.float)
        for b in range(B):
            row = shift_labels[b]                           # (L,)
            valid = row[mask[b]]
            counts = torch.bincount(valid, minlength=vocab_size).float()  # (V,)
            tf_weights[b] = counts[row]

        # --- DF: number of buffer sequences containing each unique batch token ---
        # active_buf: (N, L), unique_tokens: (U,)
        # Presence matrix: (N, L, 1) == (1, 1, U) -> any over L -> (N, U)
        unique_tokens = torch.unique(shift_labels[mask])    # (U,)
        buf_presence = (
            active_buf.unsqueeze(2) == unique_tokens.view(1, 1, -1)
        ).any(dim=1).float()                                # (N, U)
        df = buf_presence.sum(dim=0)                        # (U,)

        # --- Smoothed IDF: log((1+N)/(1+df)) + 1 ---
        idf = torch.log((1.0 + N) / (1.0 + df)) + 1.0     # (U,)

        # Map IDF back to every token position via a vocab-size lookup table
        idf_lookup = torch.zeros(vocab_size, device=device, dtype=torch.float)
        idf_lookup[unique_tokens] = idf
        idf_weights = idf_lookup[shift_labels]              # (B, L)

        # --- Normalize to unit mean over valid tokens ---
        raw_weights = tf_weights * idf_weights              # (B, L)
        avg_weight = raw_weights[mask].sum() / (num_valid + 1e-8)
        normalized_weights = raw_weights / (avg_weight + 1e-8)

        # --- Weighted loss ---
        weighted_loss = (token_losses * normalized_weights * mask).sum() / (num_valid + 1e-8)

        return (weighted_loss, outputs) if return_outputs else weighted_loss
