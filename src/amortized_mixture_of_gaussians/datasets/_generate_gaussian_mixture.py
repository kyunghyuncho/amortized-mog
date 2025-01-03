import torch
from torch.distributions import MultivariateNormal


def generate_gaussian_mixture(
    batch_size,
    minimum_components,
    maximum_components,
    dim_output,
    minimum_distance,
    minimum_log_variance,
    maximum_log_variance,
    num_samples,
):
    components = torch.randint(minimum_components, maximum_components + 1, (batch_size,))

    means = torch.zeros(batch_size, maximum_components, dim_output)

    log_variances = torch.zeros(batch_size, maximum_components, dim_output)

    existences = torch.zeros(batch_size, maximum_components)

    for i in range(batch_size):
        k = components[i].item()
        current_means = []
        for j in range(k):
            while True:
                mean = torch.randn(dim_output) * 5
                valid = True

                for existing_mean in current_means:
                    if torch.norm(mean - existing_mean) < minimum_distance:
                        valid = False
                        break

                if valid:
                    current_means.append(mean)
                    means[i, j] = mean
                    log_variances[i, j] = (
                            torch.rand(dim_output) * (maximum_log_variance - minimum_log_variance) + minimum_log_variance
                    )
                    existences[i, j] = 1
                    break

    samples = []

    for i in range(batch_size):
        k = components[i].item()
        mog_samples = []

        for j in range(k):
            mog_samples = [
                *mog_samples,
                MultivariateNormal(
                    means[i, j],
                    covariance_matrix=torch.diag(torch.exp(log_variances[i, j])),
                ).sample(
                    [
                        num_samples // k,
                    ],
                ),
            ]

        mog_samples = torch.concatenate(
            mog_samples,
            dim=0,
        )

        if num_samples % k != 0:
            mog_samples = torch.concatenate(
                [
                    mog_samples,
                    MultivariateNormal(
                        means[i, 0],
                        covariance_matrix=torch.diag(
                            torch.exp(log_variances[i, 0]),
                        ),
                    ).sample(
                        [
                            num_samples - mog_samples.shape[0],
                        ],
                    ),
                ],
                dim=0,
            )

        samples.append(mog_samples)

    samples = torch.stack(samples)

    parameters = {
        "existence": existences,
        "logvars": log_variances,
        "means": means,
        "num_components": components,
    }

    return parameters, samples
