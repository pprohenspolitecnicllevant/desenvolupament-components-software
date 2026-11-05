"""Entrena i desa les dues CNN de CIFAR-10 que el NB 6.3 posa a prova.

Una s'entrena amb data augmentation i l'altra sense. La plantilla i les dades surten de
les cel·les del NB 6.2, perquè siguin exactament les que veu l'alumnat.

Ús: python train_models.py [--epochs 25]
"""
import argparse
from pathlib import Path

import nbformat
import torch
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
OPTIONS = dict(in_channels=3, n_classes=10, image_size=32, filters=(16, 32, 64), batch_norm=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    env = {}
    notebook = nbformat.read(HERE / "NB_6_2_el_laboratori.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code" and "No cal modificar-la" in cell.source:
            exec(cell.source, env)

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
