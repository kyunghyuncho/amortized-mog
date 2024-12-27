import torch
from modules import SetTransformer2
from amortized_mog import ConditionalTransformerLM
import pytorch_lightning as pl
from trainer import MoGTrainer

def infer_mog(checkpoint_file, 
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
    trainer = MoGTrainer.load_from_checkpoint(checkpoint_file)
    set_transformer = trainer.set_transformer
    conditional_lm = trainer.conditional_lm

    # Set models to eval mode
    set_transformer.eval()
    conditional_lm.eval()

    # Add a batch dimension to the input set
    input_set = input_set.unsqueeze(0)  # Shape becomes [1, num_samples, dim_input]

    # Pass input through SetTransformer++
    with torch.no_grad():
        set_transformer_output = set_transformer(input_set)

        # Pass SetTransformer++ output through ConditionalTransformerLM for inference
        existence, means, logvars = conditional_lm(set_transformer_output)

    # Convert logits to probabilities
    existence = existence.squeeze(0)  # Remove batch dimension

    # Determine the number of components based on existence probabilities
    num_components = int(existence.sum().item())

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
    predicted_mog = infer_mog("/Users/kyunghyuncho/Repos/amortized-mog/amortized-mog-fitting/tgcpnkst/checkpoints/epoch=35-step=11268.ckpt",
                              input_set,
                              max_components=5,
                              dim_hidden=64,
                              dim_output=2,
                              num_heads=4,
                              num_blocks=2)
    
    # Print the predicted MoG parameters
    print(predicted_mog)

    # Plot `input_set` and the predicted MoG components.
    # Make sure it's pretty and saved into a .png file.
    import matplotlib.pyplot as plt
    import numpy as np

    plt.scatter(input_set[:, 0], input_set[:, 1], c='b', alpha=0.5, label='Input Set')
    for i in range(predicted_mog["num_components"]):
        mean = predicted_mog["means"][i]
        logvar = predicted_mog["logvars"][i]
        std = np.sqrt(np.exp(logvar))

        # std is a 2-dim vector. We need to plot a elipse with radius std[0] and std[1] centered at mean.
        elipse = plt.matplotlib.patches.Ellipse(mean, std[0].item(), std[1].item(), fill=False, edgecolor='r', linestyle='--', linewidth=1.5)

        plt.gca().add_artist(elipse)
    plt.legend()
    plt.savefig("predicted_mog.png")


    
    