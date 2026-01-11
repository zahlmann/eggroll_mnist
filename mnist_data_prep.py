import os
import gzip
import urllib.request
import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def download_mnist(data_dir="mnist_raw"):
    os.makedirs(data_dir, exist_ok=True)
    for fname in FILES.values():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(BASE_URL + fname, path)

def load_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # Reshape to (N, 784) and normalize to float32 [0.0, 1.0]
    return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

def load_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int32)

def prepare_mnist():
    download_mnist()
    
    print("Processing images and labels...")
    X_train = load_images(os.path.join("mnist_raw", FILES["train_images"]))
    X_test  = load_images(os.path.join("mnist_raw", FILES["test_images"]))
    
    y_train = load_labels(os.path.join("mnist_raw", FILES["train_labels"]))
    y_test  = load_labels(os.path.join("mnist_raw", FILES["test_labels"]))
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    print(f"Saving to mnist_prepped_float.npz...")
    np.savez_compressed(
        "mnist_prepped_float.npz",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test
    )
    print("Done!")

if __name__ == "__main__":
    prepare_mnist()
