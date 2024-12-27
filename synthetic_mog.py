import torch

def generate_gaussian_mixture(batch_size, min_components, max_components, dim_output, min_dist, min_logvar, max_logvar, num_samples):
    """
    Generates synthetic Gaussian Mixture Model data.

    Args:
        batch_size: Number of MoGs to generate.
        min_components: Minimum number of components in a MoG.
        max_components: Maximum number of components in a MoG.
        dim_output: Dimensionality of the Gaussian components.
        min_dist: Minimum distance between component means.
        min_logvar: Minimum log-variance for a component.
        max_logvar: Maximum log-variance for a component.
        num_samples: Number of samples to draw from each MoG.

    Returns:
        mog_params: Dictionary containing:
                     - num_components: Number of components in each MoG, shape [batch_size].
                     - means: Means of the components, shape [batch_size, max_components, dim_output].
                     - logvars: Log-variances of the components, shape [batch_size, max_components, dim_output].
        samples: Samples drawn from each MoG, shape [batch_size, num_samples, dim_output].
        existence: Binary matrix indicating the existence of each component, shape [batch_size, max_components].
    """

    num_components = torch.randint(min_components, max_components + 1, (batch_size,))
    means = torch.zeros(batch_size, max_components, dim_output)
    logvars = torch.zeros(batch_size, max_components, dim_output)
    existence = torch.zeros(batch_size, max_components)

    for i in range(batch_size):
        k = num_components[i].item()
        current_means = []

        for j in range(k):
            while True:
                # Sample a mean
                mean = torch.randn(dim_output) * 5  # Adjust scaling as needed

                # Check distance from existing means
                valid = True
                for existing_mean in current_means:
                    if torch.norm(mean - existing_mean) < min_dist:
                        valid = False
                        break

                if valid:
                    current_means.append(mean)
                    means[i, j] = mean
                    logvars[i, j] = torch.rand(dim_output) * (max_logvar - min_logvar) + min_logvar
                    existence[i, j] = 1
                    break

    # Generate samples from the MoGs
    samples = []
    for i in range(batch_size):
        k = num_components[i].item()
        mog_samples = []
        for j in range(k):
            cov = torch.diag(torch.exp(logvars[i, j]))
            dist = torch.distributions.MultivariateNormal(means[i, j], covariance_matrix=cov)
            mog_samples.append(dist.sample((num_samples // k,)))  # Distribute samples among components

        mog_samples = torch.cat(mog_samples, dim=0)

        # Add extra samples if num_samples is not divisible by k
        if num_samples % k != 0:
            extra_samples_needed = num_samples - mog_samples.shape[0]
            dist = torch.distributions.MultivariateNormal(means[i, 0], covariance_matrix=torch.diag(torch.exp(logvars[i, 0])))
            extra_samples = dist.sample((extra_samples_needed,))
            mog_samples = torch.cat([mog_samples, extra_samples], dim=0)

        samples.append(mog_samples)

    samples = torch.stack(samples)

    mog_params = {
        "num_components": num_components,
        "means": means,
        "logvars": logvars,
        "existence": existence
    }

    return mog_params, samples