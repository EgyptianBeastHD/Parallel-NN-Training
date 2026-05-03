#!/usr/bin/env python3
"""Download MNIST and save as CSV files in data/."""
import gzip
import os
import struct
import urllib.request

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download(filename: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  {dest} already exists, skipping.")
        return
    for mirror in MIRRORS:
        url = mirror + filename
        try:
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:
            print(f"  Failed ({e}), trying next mirror...")
    raise RuntimeError(f"Could not download {filename} from any mirror.")


def load_images(path: str):
    with gzip.open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read()
    import array
    data = array.array("B", buf)
    images = []
    stride = rows * cols
    for i in range(num):
        images.append(data[i * stride : (i + 1) * stride])
    return images


def load_labels(path: str):
    with gzip.open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        buf = f.read()
    import array
    return list(array.array("B", buf)[:num])


def save_csv(path: str, labels, images) -> None:
    print(f"  Saving {path} ({len(labels)} rows) ...")
    with open(path, "w") as f:
        for label, pixels in zip(labels, images):
            row = str(label) + "," + ",".join(map(str, pixels))
            f.write(row + "\n")


os.makedirs("data", exist_ok=True)

print("Downloading MNIST files...")
download("train-images-idx3-ubyte.gz", "data/train-images-idx3-ubyte.gz")
download("train-labels-idx1-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
download("t10k-images-idx3-ubyte.gz",  "data/t10k-images-idx3-ubyte.gz")
download("t10k-labels-idx1-ubyte.gz",  "data/t10k-labels-idx1-ubyte.gz")

print("\nParsing and writing CSVs...")
save_csv("data/mnist_train.csv",
         load_labels("data/train-labels-idx1-ubyte.gz"),
         load_images("data/train-images-idx3-ubyte.gz"))

save_csv("data/mnist_test.csv",
         load_labels("data/t10k-labels-idx1-ubyte.gz"),
         load_images("data/t10k-images-idx3-ubyte.gz"))

print("\nDone. Files written:")
for name in ("data/mnist_train.csv", "data/mnist_test.csv"):
    size_mb = os.path.getsize(name) / 1e6
    print(f"  {name}  ({size_mb:.1f} MB)")
