"""Static figures shown in NB 1.1 as images, so the notebook needs no code to draw them.

Run from this folder with `python make_figures.py`. The notebook loads the PNG files
by raw GitHub URL, so push them after regenerating.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def activation_functions():
    """The four activations (top) and their slope (bottom): how much the output moves when z moves."""
    z = np.linspace(-5, 5, 400)
    activations = {
        "step":    np.where(z >= 0, 1, 0),
        "sigmoid": 1 / (1 + np.exp(-z)),
        "tanh":    np.tanh(z),
        "ReLU":    np.maximum(0, z),
    }

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for col, (name, value) in enumerate(activations.items()):
        slope = np.round(np.gradient(value, z), 6)
        slope[np.abs(slope) > 5] = np.nan            # hide the vertical jump of the step at z = 0
        axes[0, col].plot(z, value, linewidth=2.5)
        axes[0, col].set_title(name)
        axes[1, col].plot(z, slope, linewidth=2.5, color="tab:orange")
        axes[1, col].set_xlabel("z")
    axes[0, 0].set_ylabel("output")
    axes[1, 0].set_ylabel("slope")
    plt.tight_layout()
    fig.savefig("activation_functions.png", dpi=100)
    plt.close(fig)


def gradient_descent_valley():
    """Gradient descent on the loss of y = w·x, from w = -0.5, with four learning rates."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 50)
    y = 2 * x + rng.normal(0, 0.2, 50)

    def loss(w):
        return np.mean((w * x - y) ** 2)

    def slope(w, step=1e-4):
        return (loss(w + step) - loss(w - step)) / (2 * step)

    w_values = np.linspace(-1, 5, 200)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, lr in zip(axes, [0.05, 0.6, 2.7, 3.2]):
        path = [-0.5]
        for _ in range(12):
            path.append(path[-1] - lr * slope(path[-1]))
        ax.plot(w_values, [loss(w) for w in w_values], linewidth=2, color="lightgray")
        ax.plot(path, [loss(w) for w in path], "o-", color="tab:red", markersize=5)
        ax.set_xlim(-1, 5)
        ax.set_ylim(0, 6)
        ax.set_title(f"learning_rate = {lr}  ->  w after 12 steps = {path[-1]:.2f}")
        ax.set_xlabel("w")
    axes[0].set_ylabel("loss")
    plt.tight_layout()
    fig.savefig("gradient_descent_valley.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    activation_functions()
    gradient_descent_valley()
