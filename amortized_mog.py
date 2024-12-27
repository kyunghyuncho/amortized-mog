import torch
import torch.nn as nn
from torch.nn import functional as F
from modules import SetTransformer2

class AmortizedMoG(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, num_heads, num_blocks, max_components):
        super().__init__()
        self.max_components = max_components
        self.set_transformer = SetTransformer2(dim_input, dim_hidden, num_heads, num_blocks, dim_output)
        self.pre_existence = nn.Linear(dim_hidden, dim_hidden)
        self.pre_mean = nn.Linear(dim_hidden, dim_hidden)
        self.pre_logvar = nn.Linear(dim_hidden, dim_hidden)
        self.existence_predictor = nn.Linear(dim_hidden, 1)  # Binary existence prediction
        self.mean_predictor = nn.Linear(dim_hidden, dim_output)  # Mean prediction
        self.logvar_predictor = nn.Linear(dim_hidden, dim_output)  # Log-variance prediction

    def forward(self, x):
        """
        Args:
            x: Input set of vectors, shape [batch_size, num_samples, dim_input]

        Returns:
            existence_logits: Logits for the existence of each component, shape [batch_size, max_components]
            means: Predicted means, shape [batch_size, max_components, dim_output]
            logvars: Predicted log-variances, shape [batch_size, max_components, dim_output]
        """
        batch_size = x.shape[0]
        # Pass the set through the Set Transformer
        set_embedding = self.set_transformer(x)  # Shape: [batch_size, dim_hidden]

        # Autoregressive prediction of Gaussian components
        existence_logits = []
        means = []
        logvars = []

        # Create a tensor to store previously predicted components
        prev_components = torch.zeros(batch_size, 0, set_embedding.shape[-1], device=x.device)

        for _ in range(self.max_components):
            # Concatenate set embedding with embeddings of previously predicted components
            combined_embedding = torch.cat([set_embedding.unsqueeze(1), prev_components], dim=1)

            # Pass the combined embedding through separate linear layers
            existence_embedding = F.relu(self.pre_existence(combined_embedding.mean(dim=1)))
            mean_embedding = F.relu(self.pre_mean(combined_embedding.mean(dim=1)))
            logvar_embedding = F.relu(self.pre_logvar(combined_embedding.mean(dim=1)))

            # Predict existence, mean, and log-variance
            existence_logit = self.existence_predictor(existence_embedding).squeeze(-1)
            mean = self.mean_predictor(mean_embedding)
            logvar = self.logvar_predictor(logvar_embedding)

            # Store predictions
            existence_logits.append(existence_logit)
            means.append(mean)
            logvars.append(logvar)

            # Create embedding for this component
            component_embedding = torch.cat([existence_logit.unsqueeze(-1), mean, logvar], dim=-1)

            # If predicted to exist, add to previous components
            # If not, we still add it but it will be weighted less in future predictions
            # due to the existence logit being small (or negative).
            prev_components = torch.cat([prev_components, component_embedding.unsqueeze(1)], dim=1)

        # Stack outputs
        existence_logits = torch.stack(existence_logits, dim=1)  # Shape: [batch_size, max_components]
        means = torch.stack(means, dim=1)  # Shape: [batch_size, max_components, dim_output]
        logvars = torch.stack(logvars, dim=1)  # Shape: [batch_size, max_components, dim_output]

        return existence_logits, means, logvars