import torch
from amortized_mog import AmortizedMoG
from modules import SetTransformer2

def infer_mog(model_checkpoint_path, input_set):
    """
    Performs inference using a trained AmortizedMoG model.

    Args:
        model_checkpoint_path: Path to the saved model checkpoint.
        input_set: Input set of vectors, shape [num_samples, dim_input].

    Returns:
        predicted_mog: Dictionary containing predicted MoG parameters:
                       - num_components: Predicted number of components.
                       - means: Predicted means of the components.
                       - logvars: Predicted log-variances of the components.
    """
    # Load the trained model
    model = AmortizedMoG.load_from_checkpoint(model_checkpoint_path)
    model.eval()

    # Add a batch dimension to the input set
    input_set = input_set.unsqueeze(0)  # Shape becomes [1, num_samples, dim_input]

    # Perform inference
    with torch.no_grad():
        existence_logits, means, logvars = model(input_set)

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
    predicted_mog = infer_mog("path/to/your/model_checkpoint.ckpt", input_set)

    # Print the predicted MoG parameters
    print(predicted_mog)