import torch
import torch.nn as nn
from torch.nn import functional as F
from modules import MDN
import math

class ConditionalTransformerLM(nn.Module):
    def __init__(self, 
                 dim_set_output, 
                 dim_output, 
                 dim_hidden, 
                 num_heads, 
                 num_blocks, 
                 max_components, 
                 mdn_components):
        super().__init__()
        self.max_components = max_components
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        # Input embedding layer
        self.input_embedding = nn.Linear(dim_set_output + 1 + 2 * dim_output, dim_hidden) # +1 for existence, +2*dim_output for mean and logvar

        # Transformer (decoder-only)
        decoder_layer = nn.TransformerEncoderLayer(d_model=dim_hidden, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=dim_hidden, 
                                                   batch_first=True,
                                                   dropout=0.1)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_blocks)

        # Output layers
        self.existence_predictor = nn.Linear(dim_hidden, 1)
        self.mean_predictor = MDN(dim_hidden, dim_output, mdn_components)
        self.logvar_predictor = MDN(dim_hidden, dim_output, mdn_components)

        # Token to indicate start of sequence (learnable)
        self.sos_token = nn.Parameter(torch.randn(1, 1, 1 + 2 * dim_output))

        # Fixed positional encoding
        self.positional_encoding = self.create_fixed_pos_encoding(3 * max_components, 
                                                                  dim_hidden) # +1 for set_transformer output

    def create_fixed_pos_encoding(self, max_len, d_model):
        """
        Creates a fixed positional encoding (sine/cosine).
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, set_transformer_output, targets=None, argmax=True):
        """
        Args:
            set_transformer_output: Output from SetTransformer++, shape [batch_size, dim_set_output]
            targets: During training, the ground truth targets (existence, mean, logvar),
                     shape [batch_size, max_components, 1 + 2*dim_output].
                     During inference, this can be None.

        Returns:
            existence_logits: Logits for component existence, shape [batch_size, max_components, 1]
            means: Predicted means, shape [batch_size, max_components, dim_output]
            logvars: Predicted logvars, shape [batch_size, max_components, dim_output]
        """

        batch_size = set_transformer_output.shape[0]

        # Create SOS token (start of sequence)
        sos_tokens = self.sos_token.repeat(batch_size, 1, 1)  # Shape: [batch_size, 1, 1 + 2 * dim_output]

        # Prepare input embeddings for the Transformer
        if targets is not None:  # Training mode
            # Concatenate Set Transformer output with target components
            set_transformer_output_expanded = set_transformer_output.unsqueeze(1).repeat(1, self.max_components+1, 1)
            expanded_targets = torch.cat([sos_tokens, targets], dim=1)
            inputs = torch.cat([set_transformer_output_expanded, expanded_targets], dim=-1)
            inputs = self.input_embedding(inputs)

            # Add positional encodings
            # inputs = torch.cat([sos_tokens, inputs], dim=1)
            inputs = inputs + self.positional_encoding[:, :inputs.shape[1], :]

            # Create target mask for autoregressive prediction
            tgt_mask = self.generate_square_subsequent_mask(inputs.shape[1]).to(inputs.device)

            # Pass through Transformer
            transformer_output = self.transformer(inputs, mask=tgt_mask)

            # Predict existence, mean, and logvar
            existence_logits = self.existence_predictor(transformer_output[:, :-1, :])  # Shape: [batch_size, max_components, 1]
            means = self.mean_predictor(transformer_output[:, :-1, :])  # Shape: 3x [batch_size, max_components, dim_output]
            logvars = self.logvar_predictor(transformer_output[:, :-1, :])  # Shape: 3x [batch_size, max_components, dim_output]
        else:  # Inference mode
            # in the inference mode, we need to collect existence, mean, and logvar on the fly
            existence_logits = None
            means = None
            logvars = None

            transformer_output = self.input_embedding(torch.cat([set_transformer_output.unsqueeze(1), 
                                                                 sos_tokens], dim=-1))
            # Add positional encodings
            transformer_output = transformer_output + self.positional_encoding[:, :1, :]
            for _ in range(self.max_components):
                # Create target mask for autoregressive prediction
                tgt_mask = self.generate_square_subsequent_mask(transformer_output.shape[1]).to(transformer_output.device)

                # Pass through Transformer
                transformer_output_step = self.transformer(transformer_output, mask=tgt_mask)

                # Take the output for the last token
                transformer_output_step = transformer_output_step[:, -1, :].unsqueeze(1)

                # Predict the next component
                existence_logit = self.existence_predictor(transformer_output_step)
                if argmax:
                    existence = (existence_logit > 0).float() # argmax. TODO: sampling
                    mean = self.mean_predictor.sample(transformer_output_step, argmax=True)
                    logvar = self.logvar_predictor.sample(transformer_output_step, argmax=True)
                else:
                    existence = (torch.sigmoid(existence_logit) > torch.rand(1)).float()
                    mean = self.mean_predictor.sample(transformer_output_step, argmax=False)
                    logvar = self.logvar_predictor.sample(transformer_output_step, argmax=False)

                # Concatenate to the existing predictions
                if existence_logits is None:
                    existence_logits = existence
                    means = mean
                    logvars = logvar
                else:
                    existence_logits = torch.cat([existence_logits, existence], dim=1)
                    means = torch.cat([means, mean], dim=1)
                    logvars = torch.cat([logvars, logvar], dim=1)

                # Form the next input token
                next_token_embedding = self.input_embedding(torch.cat([set_transformer_output.unsqueeze(1), 
                                                                       existence, mean, logvar], dim=-1))

                # Add positional encodings
                next_token_embedding = next_token_embedding + self.positional_encoding[:, transformer_output.shape[1], :].unsqueeze(1)

                # Concatenate to the transformer output
                transformer_output = torch.cat([transformer_output, next_token_embedding], dim=1)

        return existence_logits, means, logvars

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask