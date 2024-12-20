import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

def black_box_function(x):
    """Black-box function to optimize (example)."""
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))

# Define the search space
search_space = [Real(0.0, 1.0, name='x')]

# Use Bayesian optimization to minimize the negative black-box function
@use_named_args(search_space)
def objective(**params):
    return -black_box_function([params['x']])

# Perform Bayesian Optimization
result = gp_minimize(
    func=objective,
    dimensions=search_space,
    acq_func='EI',  # Expected Improvement
    n_calls=20,
    n_initial_points=5,
    random_state=42
)

# Visualize the optimization process
x = np.linspace(0.0, 1.0, 1000).reshape(-1, 1)
y = -np.array([black_box_function(xi) for xi in x])

plt.figure(figsize=(15, 5))
for i, (iteration, x_opt) in enumerate(zip(result.models, result.x_iters)):
    if i >= 5:
        break

    model = iteration
    mean, std = model.predict(x, return_std=True)

    plt.subplot(1, 5, i + 1)
    plt.plot(x, -y, 'k--', label="Ground truth")
    plt.plot(x, -mean, 'b-', label="Model")
    plt.fill_between(x.flatten(), -mean - std, -mean + std, alpha=0.2, color="blue")
    plt.scatter(result.x_iters[:i + 1], -np.array(result.func_vals[:i + 1]), c="green", label="Data")
    plt.scatter([x_opt], [result.func_vals[i]], marker="v", color="orange", label="max{acq}")
    plt.title(f"Step {i + 1}")
    plt.xlabel("x")
    plt.ylabel("f")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()
