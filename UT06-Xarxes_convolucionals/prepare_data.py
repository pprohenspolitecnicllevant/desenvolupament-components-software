"""Genera els subconjunts de dades de la UT6 a partir de les fonts originals.

Ús: python prepare_data.py CARPETA_DE_DESCARREGA

No fa falta executar-lo per seguir els notebooks: els fitxers resultants ja són al
repositori. CIFAR-10 no hi és perquè no declara cap llicència de redistribució; els
notebooks el baixen de la font oficial amb torchvision.
"""
import gzip
import io
import random
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
HERE = Path(__file__).resolve().parent

FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FASHION_CLASSES = ["samarreta", "pantalons", "jersei", "vestit", "abric",
                   "sandàlia", "camisa", "sabatilla", "bossa", "botina"]

QUICKDRAW_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
QUICKDRAW_CLASSES = {
    "apple": "poma", "banana": "plàtan", "bicycle": "bicicleta", "bird": "ocell", "book": "llibre",
    "butterfly": "papallona", "cake": "pastís", "car": "cotxe", "cat": "gat", "chair": "cadira",
    "clock": "rellotge", "cloud": "núvol", "cup": "tassa", "dog": "gos", "door": "porta",
    "eye": "ull", "fish": "peix", "flower": "flor", "guitar": "guitarra", "hat": "barret",
    "house": "casa", "key": "clau", "moon": "lluna", "mountain": "muntanya", "pizza": "pizza",
    "rainbow": "arc de Sant Martí", "star": "estrella", "sun": "sol", "tree": "arbre", "umbrella": "paraigua",
}
QUICKDRAW_TRAIN, QUICKDRAW_TEST = 2500, 500

RPS_TRAIN_URL = "https://storage.googleapis.com/learning-datasets/rps.zip"
RPS_TEST_URL = "https://storage.googleapis.com/learning-datasets/rps-test-set.zip"
RPS_CLASSES = {"rock": "pedra", "paper": "paper", "scissors": "tisora"}
RPS_SIZE, RPS_N_TRAIN, RPS_N_TEST = 160, 150, 70

EUROSAT_URL = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip"
EUROSAT_CLASSES = {
    "AnnualCrop": "conreu_anual", "Forest": "bosc", "HerbaceousVegetation": "herba", "Highway": "autopista",
    "Industrial": "industrial", "Pasture": "pastura", "PermanentCrop": "conreu_permanent",
    "Residential": "residencial", "River": "riu", "SeaLake": "mar_o_llac",
}
EUROSAT_N_TRAIN, EUROSAT_N_TEST = 80, 20


def download(url, folder):
    """Baixa un fitxer a la carpeta, si encara no hi és, i en retorna el camí."""
    path = folder / url.rsplit("/", 1)[-1]
    if not path.exists():
        print("Baixant", url)
        urllib.request.urlretrieve(url, path)
    return path


def read_idx(path):
    """Llegeix un fitxer IDX comprimit (el format de MNIST) com a array de NumPy."""
    with gzip.open(path, "rb") as file:
        _, _, dims = struct.unpack(">HBB", file.read(4))
        shape = struct.unpack(">" + "I" * dims, file.read(4 * dims))
        return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)


def prepare_fashion(folder):
    """Desa Fashion-MNIST sencer en el mateix format que el MNIST de la UT5."""
    def part(name):
        return read_idx(download(FASHION_URL + name, folder))
    target = HERE / "fashion_mnist"
    target.mkdir(exist_ok=True)
    np.savez_compressed(target / "fashion_mnist.npz",
                        X_train=part("train-images-idx3-ubyte.gz").reshape(-1, 784),
                        y_train=part("train-labels-idx1-ubyte.gz"),
                        X_test=part("t10k-images-idx3-ubyte.gz").reshape(-1, 784),
                        y_test=part("t10k-labels-idx1-ubyte.gz"),
                        class_names=np.array(FASHION_CLASSES))


def read_quickdraw_start(name, n_rows):
    """Baixa només les primeres n_rows files d'un .npy de 97 MB amb una petició parcial."""
    header_size = 128
    n_bytes = header_size + n_rows * 784
    request = urllib.request.Request(QUICKDRAW_URL + name.replace(" ", "%20") + ".npy",
                                     headers={"Range": f"bytes=0-{n_bytes - 1}"})
    with urllib.request.urlopen(request) as response:
        raw = response.read()
    version = raw[6:8]
    size = struct.unpack("<H", raw[8:10])[0] if version[0] == 1 else struct.unpack("<I", raw[8:12])[0]
    start = (10 if version[0] == 1 else 12) + size
    return np.frombuffer(raw[start:start + n_rows * 784], dtype=np.uint8).reshape(n_rows, 784)


def prepare_quickdraw():
    """Desa 30 classes de Quick Draw amb 2.500 + 500 dibuixos per classe."""
    rng = np.random.default_rng(SEED)
    parts = {"X_train": [], "y_train": [], "X_test": [], "y_test": []}
    for label, name in enumerate(QUICKDRAW_CLASSES):
        print("Quick Draw:", name)
        images = rng.permutation(read_quickdraw_start(name, QUICKDRAW_TRAIN + QUICKDRAW_TEST))
        parts["X_train"].append(images[:QUICKDRAW_TRAIN])
        parts["X_test"].append(images[QUICKDRAW_TRAIN:])
        parts["y_train"].append(np.full(QUICKDRAW_TRAIN, label, dtype=np.uint8))
        parts["y_test"].append(np.full(QUICKDRAW_TEST, label, dtype=np.uint8))
    order_train = rng.permutation(QUICKDRAW_TRAIN * len(QUICKDRAW_CLASSES))
    order_test = rng.permutation(QUICKDRAW_TEST * len(QUICKDRAW_CLASSES))
    target = HERE / "quickdraw"
    target.mkdir(exist_ok=True)
    np.savez_compressed(target / "quickdraw_30.npz",
                        X_train=np.concatenate(parts["X_train"])[order_train],
                        y_train=np.concatenate(parts["y_train"])[order_train],
                        X_test=np.concatenate(parts["X_test"])[order_test],
                        y_test=np.concatenate(parts["y_test"])[order_test],
                        class_names=np.array(list(QUICKDRAW_CLASSES.values())))


def image_names(archive, source_class):
    """Retorna les imatges d'una classe dins d'un zip, en ordre estable."""
    wanted = f"/{source_class}/"
    names = [n for n in archive.namelist() if wanted in "/" + n and n.lower().endswith((".png", ".jpg"))]
    return sorted(n for n in names if "__MACOSX" not in n)


def add_images(target, archive, names, split, label, size):
    """Afegeix imatges a un zip de sortida com a JPEG, a la carpeta split/label."""
    for index, name in enumerate(names):
        with Image.open(io.BytesIO(archive.read(name))) as image:
            image = image.convert("RGB")
            if image.size != (size, size):
                image = image.resize((size, size), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
        target.writestr(f"{split}/{label}/{index:04d}.jpg", buffer.getvalue())


def prepare_rps(folder):
    """Desa 150 + 70 imatges per classe de pedra-paper-tisora, a 160 píxels."""
    rng = random.Random(SEED)
    target_dir = HERE / "pedra_paper_tisora"
    target_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(target_dir / "rps.zip", "w") as target:
        for url, split, n_images in [(RPS_TRAIN_URL, "train", RPS_N_TRAIN), (RPS_TEST_URL, "test", RPS_N_TEST)]:
            with zipfile.ZipFile(download(url, folder)) as archive:
                for source, label in RPS_CLASSES.items():
                    names = image_names(archive, source)
                    add_images(target, archive, rng.sample(names, n_images), split, label, RPS_SIZE)


def prepare_eurosat(folder):
    """Desa 80 + 20 imatges per classe d'EuroSAT, a la seva mida original de 64 píxels."""
    rng = random.Random(SEED)
    target_dir = HERE / "eurosat"
    target_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(download(EUROSAT_URL, folder)) as archive, \
            zipfile.ZipFile(target_dir / "eurosat_1000.zip", "w") as target:
        for source, label in EUROSAT_CLASSES.items():
            names = rng.sample(image_names(archive, source), EUROSAT_N_TRAIN + EUROSAT_N_TEST)
            add_images(target, archive, names[:EUROSAT_N_TRAIN], "train", label, 64)
            add_images(target, archive, names[EUROSAT_N_TRAIN:], "test", label, 64)


if __name__ == "__main__":
    download_folder = Path(sys.argv[1])
    download_folder.mkdir(parents=True, exist_ok=True)
    prepare_fashion(download_folder)
    prepare_quickdraw()
    prepare_rps(download_folder)
    prepare_eurosat(download_folder)
