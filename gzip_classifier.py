# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "datasets==4.0.0",
#   "numpy==2.3.2"
# ]
# ///
""" 
Text classification with gzip 🗜️

Adapted from Jiang et al. (2023) with some performance improvements

References:
+ (Ken Schutte, 2023) "Gzip beats BERT?" Part 2: dataset issues, improved speed, and results
https://kenschutte.com/gzip-knn-paper2/
"""

# Standard imports
import gzip

# Library imports
import numpy as np

from datasets import load_dataset


def clen(text: str) -> int:
    """Get the compressed length of a string.
    
    Args:
        text (str): Input text string to compress
        
    Returns:
        int: Length of the gzip-compressed text in bytes
    """
    return len(gzip.compress(text.encode("utf-8")))


def ncd(cx: int, cy: int, cxy: int) -> float:
    """Computes the Normalized Compression Distance (NCD).
    
    Args:
        cx (int): Compressed length of first string
        cy (int): Compressed length of second string  
        cxy (int): Compressed length of concatenated strings
        
    Returns:
        float: Normalized compression distance between 0 and 1,
               where 0 indicates identical strings and 1 indicates
               completely different strings
    """
    return (cxy - min(cx, cy)) / max(cx, cy)


def knn(dist: np.array, train_labels: list[int], k: int = 5) -> list[int]:
    """Predict labels using k-NN with majority vote.
    
    Args:
        dist (np.array): Distance matrix of shape (n_train, n_test) where
                        dist[i, j] is the distance between training sample i
                        and test sample j
        train_labels (list[int]): List of integer labels for training samples
        k (int, optional): Number of nearest neighbors to consider. Defaults to 5.
        
    Returns:
        list[int]: Predicted labels for each test sample, determined by
                  majority vote among k nearest neighbors
    """
    preds = []
    for i in range(dist.shape[1]):
        knn_indices = np.argsort(dist[:, i])[:k]
        knn_labels = [train_labels[j] for j in knn_indices]
        preds.append(max(set(knn_labels), key=knn_labels.count))
    return preds


def main():
    """Main entrypoint for gzip-based text classification.
    
    Loads AG News dataset, computes gzip-based distances between texts,
    and performs k-NN classification using normalized compression distance.
    Prints the classification accuracy on the test set.
    """
    # Load data from HF
    train, test = load_dataset("ag_news", split=["train[:1%]", "test[:1%]"])

    # Calculate compressed lengths
    train_clens = [clen(item["text"]) for item in train]
    test_clens = [clen(item["text"]) for item in test]
    concat_clens = [[clen(item1["text"] + item2["text"]) for item2 in test] for item1 in train]

    # Compute distance matrix
    dist = np.zeros((len(train), len(test)))
    for i in range(len(train)):
        for j in range(len(test)):
            dist[i, j] = ncd(train_clens[i], test_clens[j], concat_clens[i][j])

    # Predict labels using k-NN
    train_labels = [item["label"] for item in train]
    test_labels = [item["label"] for item in test]
    predicted_labels = knn(dist, train_labels, k=5)
    print(f"Accuracy: {np.mean(np.array(predicted_labels) == np.array(test_labels))}")


if __name__ == "__main__":
    main()
