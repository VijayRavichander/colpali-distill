import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class ColBertPairwiseDistillLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
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
                batch_size == batch_size_c == batch_size_t == batch_size_tc and
                emb_dim == emb_dim_c == emb_dim_t == emb_dim_tc and
                num_query_tokens == num_query_tokens_t and
                num_context_tokens == num_context_tokens_t
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
