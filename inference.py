import time

import torch
from modules import SetTransformer2
from amortized_mog import ConditionalTransformerLM
import pytorch_lightning as pl
from trainer import MoGTrainer
import sklearn.mixture

from utils import find_latest_checkpoint

def infer_mog(checkpoint_file, input_set, timeit=False):
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
    if timeit:
        start_time = time.time()
    with torch.no_grad():
        set_transformer_output = set_transformer(input_set)

        # Pass SetTransformer++ output through ConditionalTransformerLM for inference
        existence, means, logvars = conditional_lm(set_transformer_output)
    if timeit:
        time_taken = time.time() - start_time
        print(f"Time taken for inference: {time_taken:.2f}s")

    # Convert logits to probabilities
    existence = existence.squeeze(0)  # Remove batch dimension

    # Determine the number of components based on existence probabilities
    num_components = int(existence.sum().item())

    predicted_mog = {
        "num_components": num_components,
        "means": means.squeeze(0)[:num_components],  # Remove batch dimension and take only existing components
        "logvars": logvars.squeeze(0)[:num_components]  # Remove batch dimension and take only existing components
    }

    if timeit:
        return predicted_mog, time_taken
    return predicted_mog

# Example Usage
if __name__ == "__main__":
    # sample input set 1
    input_set = torch.randn(100, 2) / 3.
    input_set[:, 0] = input_set[:, 0] - 3.
    input_set = torch.cat([input_set, torch.randn(100, 2) / 3. - 3.], dim=0)
    input_set = torch.cat([input_set, torch.randn(100, 2) / 3. + 3.], dim=0)
    n_true_components = 3

    # # sample input set 2
    # input_set = torch.rand(100, 2)
    # input_set[:, 0] = input_set[:, 0] * 5.
    # input_set[:, 1] = input_set[:, 1] + 2.
    # input_set = torch.cat([input_set, torch.randn(100, 2) / 3.], dim=0)
    # n_true_components = 2

    # # sample input set 3
    # input_set = torch.randn(100, 2)
    # input_set[:, 0] = input_set[:, 0] * 5.
    # input_set2 = torch.randn(100, 2)
    # input_set2[:, 1] = input_set2[:, 1] * 5.
    # input_set = torch.cat([input_set, input_set2], dim=0)
    # n_true_components = 2

    n_trials = 1
    checkpoint_path = "/Users/kyunghyuncho/Repos/amortized-mog/amortized-mog-fitting"

    times = []
    for i in range(n_trials):
        # Perform inference and time it
        predicted_mog, time_taken = infer_mog(find_latest_checkpoint(checkpoint_path+"/1m5u40lm/checkpoints/"),
                                              input_set, timeit=True)
        times.append(time_taken)
    print(f"Average time taken for {n_trials} trials: {sum(times) / n_trials:.5f}s")
    print(f"Standard deviation: {torch.tensor(times).std():.5f}s")

    # Print the predicted MoG parameters
    print(predicted_mog)

    # Fit MoG with 4 components using scikit-learn
    times = []
    for i in range(n_trials):
        start_time = time.time()
        gmm = sklearn.mixture.GaussianMixture(n_components=n_true_components, 
                                            covariance_type='diag')
        gmm.fit(input_set)
        times.append(time.time() - start_time)
    print(f"Average time taken for {n_trials} trials (scikit-learn): {sum(times) / n_trials:.5f}s")
    print(f"Standard deviation: {torch.tensor(times).std():.5f}s")

    # Plot `input_set` and the predicted MoG components.
    # Make sure it's pretty and saved into a .png file.
    import matplotlib.pyplot as plt
    import numpy as np

    # input set should be drawn with small black dots.
    plt.scatter(input_set[:, 0], input_set[:, 1], 
                c='k', label='Input Set', s=5, alpha=0.3)

    for i in range(predicted_mog["num_components"]):
        mean = predicted_mog["means"][i]
        logvar = predicted_mog["logvars"][i]
        std = np.sqrt(np.exp(logvar))

        # std is a 2-dim vector. We need to plot a elipse with radius std[0] and std[1] centered at mean.
        elipse = plt.matplotlib.patches.Ellipse(mean, 
                                                2 * std[0].item(), 
                                                2 * std[1].item(), 
                                                fill=False, 
                                                edgecolor='r', 
                                                linestyle='--', 
                                                linewidth=2.5)

        plt.gca().add_artist(elipse)
    
    for i in range(n_true_components):
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]
        std = np.sqrt(cov)

        # std is a 2-dim vector. We need to plot a elipse with radius std[0] and std[1] centered at mean.
        elipse = plt.matplotlib.patches.Ellipse(mean, 
                                                2 * std[0], 
                                                2 * std[1], 
                                                fill=False, edgecolor='b', linestyle='--', linewidth=2.5)

        plt.gca().add_artist(elipse)
    
    # put a custom legend:
    #  1. blue dot: input points
    #  2. red dashed ellipse: predicted MoG components
    #  3. green dashed ellipse: scikit-learn's MoG components
    plt.legend([plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='k', markersize=10),
                plt.Line2D([0], [0], color='r', linestyle='--', linewidth=1.5),
                plt.Line2D([0], [0], color='b', linestyle='--', linewidth=1.5)],
               ['Input Set', 'Predicted MoG', 'Scikit-Learn MoG'])
    plt.savefig("predicted_mog.png")    
    