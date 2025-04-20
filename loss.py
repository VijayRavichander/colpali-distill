import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

class ColBertPairwiseDistillLoss(torch.nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.mse_loss = MSELoss()
        self.alpha = alpha  # Weighting factor for MSE loss

    def forward(self, query_embeddings, doc_embeddings, teacher_query_outputs = None, teacher_doc_outputs = None, eval = False):
        """
        Computes the contrastive loss based on Late Interaction and adds an MSE loss for distillation.
        
        Args:
            query_embeddings (torch.Tensor): Shape (batch_size, num_query_tokens, emb_dim)
            doc_embeddings (torch.Tensor): Shape (batch_size, num_context_tokens, emb_dim)
            teacher_query_outputs (torch.Tensor): Shape (batch_size, num_query_tokens, emb_dim)
            teacher_doc_outputs (torch.Tensor): Shape (batch_size, num_context_tokens, emb_dim)
        
        Returns:
            torch.Tensor: The computed loss.
        """
        batch_size, num_query_tokens, emb_dim = query_embeddings.shape
        batch_size_c, num_context_tokens, emb_dim_c = doc_embeddings.shape

        
        if not eval:
            batch_size_t, num_query_tokens_t, emb_dim_t = teacher_query_outputs.shape
            batch_size_tc, num_context_tokens_t, emb_dim_tc = teacher_doc_outputs.shape

            assert (
                batch_size == batch_size_c == batch_size_t == batch_size_tc
            ), "Shape mismatch"

        else:
            assert batch_size == batch_size_c, "Shape mismatch"

        scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
        pos_scores = torch.diag(scores)

        mask = torch.eye(scores.shape[0], device=scores.device).bool()
        s_masked = scores.masked_fill(mask, float('-inf'))
        neg_scores = s_masked.max(dim=1)[0]
        contrastive_loss = F.softplus(neg_scores - pos_scores).mean()

        if not eval:
            # MSE loss 
            teacher_scores = torch.einsum("bnd,csd->bcns", teacher_query_outputs, teacher_doc_outputs).max(dim=3)[0].sum(dim=2)

            mse_loss = self.mse_loss(scores.to(torch.float16), teacher_scores.to(torch.float16))

            # Combine contrastive loss with MSE loss
            loss = contrastive_loss + self.alpha * mse_loss
            
        else:
            loss = contrastive_loss

        return loss

class ColBertPairwiseDistillKLLoss(nn.Module): # Renamed for clarity
    """
    Computes a loss combining ColBERT's contrastive loss with KL Divergence
    for distillation from a teacher model's late interaction scores.
    """
    def __init__(self, alpha=0.5, temperature=1.0): # Adjusted default alpha, added temperature
        super().__init__()
        # KLDivLoss expects log-probabilities as input and probabilities as target
        # reduction='batchmean' averages the loss per sample in the batch
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=False)
        self.alpha = alpha  # Weighting factor for KL divergence loss
        self.temperature = temperature # Temperature for softening distributions

    def forward(self, query_embeddings, doc_embeddings, teacher_query_outputs=None, teacher_doc_outputs=None, eval=False):
        """
        Computes the combined loss.

        Args:
            query_embeddings (torch.Tensor): Student query embeddings. Shape (batch_size, num_query_tokens, emb_dim)
            doc_embeddings (torch.Tensor): Student document embeddings. Shape (batch_size, num_doc_tokens, emb_dim)
            teacher_query_outputs (torch.Tensor, optional): Teacher query embeddings. Shape (batch_size, num_query_tokens, emb_dim). Required if eval=False.
            teacher_doc_outputs (torch.Tensor, optional): Teacher document embeddings. Shape (batch_size, num_doc_tokens, emb_dim). Required if eval=False.
            eval (bool): If True, only compute the contrastive loss. Defaults to False.

        Returns:
            torch.Tensor: The computed loss.
        """
        batch_size, num_query_tokens, emb_dim = query_embeddings.shape
        batch_size_c, num_doc_tokens, emb_dim_c = doc_embeddings.shape

        # Basic shape checks
        assert batch_size == batch_size_c, f"Batch size mismatch: {batch_size} vs {batch_size_c}"
        assert emb_dim == emb_dim_c, f"Embedding dimension mismatch: {emb_dim} vs {emb_dim_c}"

        # --- Student Score Calculation (ColBERT Late Interaction) ---
        # scores[i, j] = similarity between query i and doc j
        scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)

        # --- Contrastive Loss Calculation ---
        # Uses the student scores within the batch
        pos_scores = torch.diag(scores) # Scores between query i and doc i
        mask = torch.eye(scores.shape[0], device=scores.device, dtype=torch.bool)
        neg_scores_masked = scores.masked_fill(mask, float('-inf')) # Mask out positive pairs
        # Option 1: Max negative score per query
        neg_scores = neg_scores_masked.max(dim=1)[0]
        # Option 2: LogSumExp over negative scores per query (often more stable)
        # neg_scores = torch.logsumexp(neg_scores_masked, dim=1)

        # Using Softplus for the margin loss: log(1 + exp(neg - pos))
        # Equivalent to CrossEntropyLoss if neg_scores were logits for the negative class
        contrastive_loss = F.softplus(neg_scores - pos_scores).mean()

        # --- Distillation Loss Calculation (KL Divergence) ---
        if not eval:
            # Ensure teacher outputs are provided for training
            assert teacher_query_outputs is not None and teacher_doc_outputs is not None, \
                "Teacher outputs must be provided when eval=False"

            # Teacher shape checks
            batch_size_t, num_query_tokens_t, emb_dim_t = teacher_query_outputs.shape
            batch_size_tc, num_doc_tokens_t, emb_dim_tc = teacher_doc_outputs.shape
            assert (
                batch_size == batch_size_t == batch_size_tc and
                emb_dim == emb_dim_t == emb_dim_tc and
                num_query_tokens == num_query_tokens_t and
                num_doc_tokens == num_doc_tokens_t
            ), "Shape mismatch between student and teacher outputs"

            with torch.no_grad(): # Teacher calculations should not require gradients
                # Calculate teacher scores (late interaction)
                teacher_scores = torch.einsum("bnd,csd->bcns", teacher_query_outputs, teacher_doc_outputs).max(dim=3)[0].sum(dim=2)

            # Apply temperature scaling
            student_scores_temp = scores / self.temperature
            teacher_scores_temp = teacher_scores / self.temperature

            # Calculate student log-probabilities and teacher probabilities
            # Softmax over documents (dim=1) for each query
            student_log_probs = F.log_softmax(student_scores_temp, dim=1)
            # Detach teacher probabilities as they are the target distribution
            teacher_probs = F.softmax(teacher_scores_temp, dim=1) # No detach needed due to torch.no_grad() context

            # Calculate KL divergence loss
            # Measures how much student distribution diverges from teacher's
            kl_div_loss = self.kl_loss(student_log_probs, teacher_probs)

            # Scale KL loss by T^2 - standard practice when using temperature in KL distillation
            # This accounts for the softening effect of T on the gradients
            kl_div_loss = kl_div_loss * (self.temperature ** 2)

            # Combine contrastive loss with KL divergence loss
            loss = contrastive_loss + self.alpha * kl_div_loss

        else: # If eval=True, only return the contrastive loss
            loss = contrastive_loss

        return loss
