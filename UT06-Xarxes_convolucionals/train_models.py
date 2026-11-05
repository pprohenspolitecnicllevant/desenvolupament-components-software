"""Entrena i desa les dues CNN de CIFAR-10 que el NB 6.3 posa a prova.

Una s'entrena amb data augmentation i l'altra sense. La plantilla i les dades surten de
les cel·les del NB 6.2, perquè siguin exactament les que veu l'alumnat.

Ús: python train_models.py [--epochs 25]
    python train_models.py --export-json pesos.json    (la CNN de MNIST de l'animació)
"""
import argparse
import json
from pathlib import Path

import nbformat
import torch
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
OPTIONS = dict(in_channels=3, n_classes=10, image_size=32, filters=(16, 32, 64), batch_norm=True)


def export_json(env, target):
    """Entrena la CNN de MNIST de l'animació i en desa els pesos amb 3 decimals."""
    train_set, val_set, _, _ = env["load_dataset"]("mnist")
    model = env["make_cnn"](1, 10)
    train_loader, val_loader = env["make_loaders"](train_set, val_set)
    env["train_model"](model, train_loader, val_loader, epochs=3, lr=0.001)
    state = model.cpu().state_dict()
    round3 = lambda tensor: [round(value, 3) for value in tensor.flatten().tolist()]
    layers = {"conv1": "0", "conv2": "3", "fc1": "7", "fc2": "9"}
    weights = {name: {"w": round3(state[f"{position}.weight"]), "b": round3(state[f"{position}.bias"])}
               for name, position in layers.items()}
    images, labels = val_set.tensors
    samples = [(images[labels == digit][0].flatten() * 255).round().int().tolist() for digit in range(10)]
    Path(target).write_text(json.dumps({"weights": weights, "samples": samples}, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--export-json", help="on desar els pesos de la CNN de l'animació")
    args = parser.parse_args()

    env = {}
    notebook = nbformat.read(HERE / "NB_6_2_el_laboratori.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code" and "No cal modificar-la" in cell.source:
            exec(cell.source, env)

    if args.export_json:
        export_json(env, args.export_json)
        return

    train_set, val_set, _, _ = env["load_dataset"]("cifar10")
    flip_and_crop = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomCrop(32, padding=4)])
    (HERE / "models").mkdir(exist_ok=True)
    for name, augment in [("cifar_cnn", None), ("cifar_cnn_aug", flip_and_crop)]:
        model = env["make_cnn"](**OPTIONS)
        train_loader, val_loader = env["make_loaders"](train_set, val_set, augment=augment)
        history = env["train_model"](model, train_loader, val_loader, epochs=args.epochs, lr=0.001)
        torch.save(model.cpu().state_dict(), HERE / "models" / f"{name}.pt")
        print(f"{name}: {env['count_parameters'](model)} pesos, encert de validació {history['val_acc'][-1]:.1%}")


if __name__ == "__main__":
    main()
