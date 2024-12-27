import torch
from modules import SetTransformer2
from amortized_mog import ConditionalTransformerLM
import pytorch_lightning as pl

def infer_mog(set_transformer_checkpoint, 
              conditional_lm_checkpoint, 
              input_set, 
              max_components, 
              dim_hidden, 
              dim_output, 
              num_heads, 
              num_blocks):
    """
    Performs inference using a trained SetTransformer++ and ConditionalTransformerLM.

    Args:
        set_transformer_checkpoint: Path to the saved SetTransformer++ checkpoint.
        conditional_lm_checkpoint: Path to the saved ConditionalTransformerLM checkpoint.
        input_set: Input set of vectors, shape [num_samples, dim_input].
        max_components: Maximum number of components to predict.
        dim_hidden: Hidden dimension of the models.
        dim_output: Dimensionality of the output (Gaussian components).

    Returns:
        predicted_mog: Dictionary containing predicted MoG parameters:
                       - num_components: Predicted number of components.
                       - means: Predicted means of the components.
                       - logvars: Predicted log-variances of the components.
    """

    # Load the trained models
    set_transformer = SetTransformer2.load_from_checkpoint(set_transformer_checkpoint)
    conditional_lm = ConditionalTransformerLM.load_from_checkpoint(conditional_lm_checkpoint)

    # Set models to eval mode
    set_transformer.eval()
    conditional_lm.eval()

    # Add a batch dimension to the input set
    input_set = input_set.unsqueeze(0)  # Shape becomes [1, num_samples, dim_input]

    # Pass input through SetTransformer++
    with torch.no_grad():
        set_transformer_output = set_transformer(input_set)

        # Pass SetTransformer++ output through ConditionalTransformerLM for inference
        existence_logits, means, logvars = conditional_lm(set_transformer_output)

    # Convert logits to probabilities
    existence_probs = torch.sigmoid(existence_logits).squeeze(0)  # Remove batch dimension

    # Determine the number of components based on existence probabilities
    num_components = (existence_probs > 0.5).sum().item()

    predicted_mog = {
        "num_components": num_components,
        "means": means.squeeze(0)[:num_components],  # Remove batch dimension and take only existing components
        "logvars": logvars.squeeze(0)[:num_components]  # Remove batch dimension and take only existing components
    }

    return predicted_mog

# Example Usage
if __name__ == "__main__":
    # Sample input set (replace with your actual input data)
    input_set = torch.randn(100, 2)

    # Perform inference
    predicted_mog = infer_mog("path/to/set_transformer_checkpoint.ckpt",
                              "path/to/conditional_lm_checkpoint.ckpt",
                              input_set,
                              max_components=5,
                              dim_hidden=64,
                              dim_output=2,
                              num_heads=4,
                              num_blocks=2)

    # Print the predicted MoG parameters
    print(predicted_mog)