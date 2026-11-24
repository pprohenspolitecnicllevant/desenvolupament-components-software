"""Genera corbes_learning_rate.png, la figura de l'exercici 2 dels exercicis de consolidació de la UT5.

La mateixa xarxa (8 neurones ocultes, ReLU, gradient descent pur) entrenada sobre make_moons amb quatre
learning rates. Les corbes surten desordenades i sense el valor del learning rate, perquè l'alumnat
les ha de relacionar amb els quatre casos de la vall (secció 7 del NB 1.1). Les respostes són al
solucionari del repositori privat.

Execució: python make_figura.py
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y = make_moons(n_samples=600, noise=0.25, random_state=42)
PANELS = {"A": 10.0, "B": 0.001, "C": 30.0, "D": 0.5}   # lletra de la corba -> learning rate

plt.rcParams.update({"font.family": "Helvetica Neue", "axes.grid": True, "grid.alpha": 0.3})
fig, grafics = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for grafic, (letter, lr) in zip(grafics, PANELS.items()):
    model = MLPClassifier(hidden_layer_sizes=(8,), solver="sgd", momentum=0, learning_rate_init=lr,
                          max_iter=300, n_iter_no_change=300, random_state=42).fit(X, y)
    grafic.plot(model.loss_curve_, color="#1C7293", linewidth=1.8)
    grafic.set_title(f"Corba {letter}", fontsize=14)
    grafic.set_xlabel("epoch")
    grafic.set_ylim(0, 2.5)
grafics[0].set_ylabel("loss d'entrenament")
fig.suptitle("La mateixa xarxa amb quatre learning rates diferents", fontsize=14)
plt.tight_layout()
fig.savefig(Path(__file__).with_name("corbes_learning_rate.png"), dpi=110)
