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


def activation_bends():
    """Two hidden neurons and an output that adds them, with the same weights: without activation
    and with sigmoid. Lines add up to a line; two soft steps add up to a bump around the middle points."""
    x = np.linspace(-0.5, 4.5, 400)
    z_a = 4 * (x - 1)                 # neuron A: switches on at x = 1
    z_b = -3 * (x - 3)                # neuron B: switches off at x = 3
    x_points = np.array([0.0, 0.4, 1.6, 2.0, 2.4, 3.6, 4.0])
    y_points = np.array([0, 0, 1, 1, 1, 0, 0])

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Columns: the activation. Rows: the hidden neurons (top) and the output that adds them (bottom)
    columns = [
        ("No activation", z_a, z_b, "line + line = line", (-8, 16), "upper center"),
        ("Sigmoid", sigmoid(z_a), sigmoid(z_b), "two soft steps = a bump", (-0.25, 1.3), "center"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for col, (name, a, b, result, ylim, legend_loc) in enumerate(columns):
        ax = axes[0, col]
        ax.plot(x, a, linewidth=2.5, color="tab:green", label="neuron A")
        ax.plot(x, b, linewidth=2.5, color="tab:purple", label="neuron B")
        ax.set_ylim(ylim)
        ax.set_title(f"{name}: hidden neurons")
        ax.legend(loc=legend_loc, framealpha=1)

        ax = axes[1, col]
        ax.plot(x, a + b - 1, linewidth=3, color="black")
        ax.scatter(x_points, np.full(len(x_points), 0.08), c=y_points, cmap="coolwarm",
                   edgecolor="k", s=70, zorder=3, transform=ax.get_xaxis_transform())
        ax.set_ylim(ylim)
        ax.set_title(f"{name}: output = A + B - 1  ->  {result}")
        ax.set_xlabel("x")
    plt.tight_layout()
    fig.savefig("activation_bends.png", dpi=100)
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


def perceptron_plane():
    """The AND perceptron (w = (1, 1), b = -1.5): the line in 2D and the tilted plane z in 3D."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    z_points = X @ np.array([1.0, 1.0]) - 1.5
    point_colors = np.where(z_points >= 0, "tab:red", "tab:blue")

    fig = plt.figure(figsize=(13, 5))

    # Left: the plane seen from above, coloured by the perceptron's answer
    ax = fig.add_subplot(1, 2, 1)
    g = np.linspace(-0.5, 1.5, 300)
    G1, G2 = np.meshgrid(g, g)
    ax.contourf(G1, G2, (G1 + G2 - 1.5 >= 0).astype(int), levels=[-0.5, 0.5, 1.5],
                colors=["#9ecae1", "#fdae6b"], alpha=0.5)
    ax.plot(g, 1.5 - g, color="black", linewidth=2.5, label="boundary: z = 0")
    ax.scatter(X[:, 0], X[:, 1], c=point_colors, edgecolor="k", s=60, zorder=3)
    for (x1, x2), zz in zip(X, z_points):
        ax.annotate(f"z = {zz:+.1f}", (x1, x2), textcoords="offset points", xytext=(8, 8))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(False)
    ax.legend(loc="lower left")
    ax.set_title("In 2D: a line splits the plane")

    # Right: z as the height of each point, a tilted plane that crosses the floor z = 0
    ax = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
    g = np.linspace(-0.25, 1.25, 31)
    G1, G2 = np.meshgrid(g, g)
    Z = G1 + G2 - 1.5
    colors = np.where(Z[..., None] >= 0, (1.0, 0.5, 0.1, 0.6), (0.1, 0.45, 0.6, 0.4))
    ax.plot_surface(G1, G2, np.zeros_like(Z), color="lightgray", alpha=0.4, shade=False)
    ax.plot_surface(G1, G2, Z, facecolors=colors, shade=False, linewidth=0)
    edge = np.linspace(0.25, 1.25, 20)
    ax.plot(edge, 1.5 - edge, 0 * edge, color="black", linewidth=2.5, label="boundary: z = 0")
    for (x1, x2), zz, color in zip(X, z_points, point_colors):
        ax.plot([x1, x1], [x2, x2], [0, zz], color="gray", linestyle=":")
        ax.scatter(x1, x2, zz, color=color, s=50, edgecolor="k", depthshade=False)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=-62)
    ax.legend(loc="upper left")
    ax.set_title("In 3D: z is a tilted plane, cut by the floor z = 0")

    plt.tight_layout()
    fig.savefig("perceptron_plane.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    perceptron_plane()
    activation_functions()
    activation_bends()
    gradient_descent_valley()
