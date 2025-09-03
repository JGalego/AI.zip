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
import logging
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from typing import NamedTuple

# Library imports
import numpy as np

from datasets import load_dataset


class CompressionData(NamedTuple):
    """Container for text compression data."""
    train_texts: list[str]
    test_texts: list[str]
    train_clens: list[int]
    test_clens: list[int]


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=10000)
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


def knn_vectorized(dist: np.ndarray, train_labels: np.ndarray, k: int = 5) -> np.ndarray:
    """Optimized k-NN prediction using vectorized operations.
    
    Args:
        dist (np.ndarray): Distance matrix of shape (n_train, n_test)
        train_labels (np.ndarray): Array of integer labels for training samples
        k (int, optional): Number of nearest neighbors to consider. Defaults to 5.
        
    Returns:
        np.ndarray: Predicted labels for each test sample
    """
    # Get k nearest neighbors for all test samples at once
    knn_indices = np.argpartition(dist, k, axis=0)[:k, :]

    # Get labels for nearest neighbors
    knn_labels = train_labels[knn_indices]

    # Vectorized majority vote using bincount
    predictions = np.zeros(dist.shape[1], dtype=int)
    for i in range(dist.shape[1]):
        predictions[i] = np.bincount(knn_labels[:, i]).argmax()

    return predictions


def compute_clens_parallel(texts: list[str], max_workers: int = None) -> list[int]:
    """Compute compressed lengths in parallel.

    Args:
        texts (list[str]): List of text strings to compress
        max_workers (int): Maximum number of worker processes

    Returns:
        list[int]: List of compressed lengths
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(texts))

    # For small datasets, use threading to avoid process overhead
    if len(texts) < 100:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(clen, texts))

    # For larger datasets, use multiprocessing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(clen, texts))


def compute_distances_adaptive(train_texts: list[str], test_texts: list[str],
                             train_clens: list[int], test_clens: list[int]) -> np.ndarray:
    """Adaptively choose the best parallelization strategy based on dataset size.

    Args:
        train_texts (list[str]): Training text samples
        test_texts (list[str]): Test text samples  
        train_clens (list[int]): Compressed lengths of training texts
        test_clens (list[int]): Compressed lengths of test texts

    Returns:
        np.ndarray: Distance matrix of shape (n_train, n_test)
    """
    n_train, n_test = len(train_texts), len(test_texts)
    total_operations = n_train * n_test

    # For small datasets, use simple threading
    if total_operations < 10000:
        logger.info("Using threaded approach for small dataset")
        return compute_distances_threaded(train_texts, test_texts, train_clens, test_clens)

    # For larger datasets, use chunked multiprocessing
    logger.info("Using chunked multiprocessing for large dataset")
    # Adaptive chunk size based on dataset size and CPU count
    optimal_chunk_size = max(5, min(100, n_train // (mp.cpu_count() * 3)))
    data = CompressionData(train_texts, test_texts, train_clens, test_clens)
    return compute_distances_chunked_parallel(data, chunk_size=optimal_chunk_size)


def compute_distances_threaded(train_texts: list[str], test_texts: list[str],
                             train_clens: list[int], test_clens: list[int],
                             max_workers: int = None) -> np.ndarray:
    """Compute distances using threading (good for smaller datasets).

    Args:
        train_texts (list[str]): Training text samples
        test_texts (list[str]): Test text samples  
        train_clens (list[int]): Compressed lengths of training texts
        test_clens (list[int]): Compressed lengths of test texts
        max_workers (int): Maximum number of worker threads

    Returns:
        np.ndarray: Distance matrix of shape (n_train, n_test)
    """
    n_train, n_test = len(train_texts), len(test_texts)
    dist = np.zeros((n_train, n_test))

    if max_workers is None:
        max_workers = min(mp.cpu_count(), n_train)

    def compute_row(i):
        row = np.zeros(n_test)
        for j in range(n_test):
            concat_len = clen(train_texts[i] + test_texts[j])
            row[j] = ncd(train_clens[i], test_clens[j], concat_len)
        return i, row

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_row, i) for i in range(n_train)]
        for future in futures:
            i, row = future.result()
            dist[i] = row

    return dist


def compute_chunk_distances(args):
    """Compute distances for a chunk of the distance matrix.

    Args:
        args: Tuple of (train_chunk_indices, train_texts, test_texts, 
                       train_clens, test_clens)

    Returns:
        tuple: (start_idx, distance_chunk)
    """
    start_idx, train_indices, train_texts, test_texts, train_clens, test_clens = args

    chunk_size = len(train_indices)
    n_test = len(test_texts)
    chunk_dist = np.zeros((chunk_size, n_test))

    for local_i, global_i in enumerate(train_indices):
        for j in range(n_test):
            concat_len = clen(train_texts[global_i] + test_texts[j])
            chunk_dist[local_i, j] = ncd(train_clens[global_i], test_clens[j], concat_len)

    return start_idx, chunk_dist


def compute_distances_chunked_parallel(data: CompressionData, *,
                                     chunk_size: int = 50, max_workers: int = None) -> np.ndarray:
    """Compute NCD distance matrix using chunked parallel processing.

    Args:
        data (CompressionData): Container with texts and compressed lengths
        chunk_size (int): Size of chunks for parallel processing
        max_workers (int): Maximum number of worker processes

    Returns:
        np.ndarray: Distance matrix of shape (n_train, n_test)
    """
    dist = np.zeros((len(data.train_texts), len(data.test_texts)))
    max_workers = max_workers or mp.cpu_count()

    # Create chunks and process in parallel
    chunks = [
        (i, list(range(i, min(i + chunk_size, len(data.train_texts)))),
         data.train_texts, data.test_texts, data.train_clens, data.test_clens)
        for i in range(0, len(data.train_texts), chunk_size)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for future in executor.map(compute_chunk_distances, chunks):
            start_idx, chunk_dist = future
            dist[start_idx:start_idx + chunk_dist.shape[0]] = chunk_dist

    return dist
def main():
    """Main entrypoint for gzip-based text classification.
    
    Loads AG News dataset, computes gzip-based distances between texts,
    and performs k-NN classification using normalized compression distance.
    Logs the classification accuracy and performance metrics.
    """
    # Load data from HF
    logger.info("Loading dataset")
    train, test = load_dataset("ag_news", split=["train[:10%]", "test[:10%]"])

    # Extract texts and labels
    train_texts = [item["text"] for item in train]
    test_texts = [item["text"] for item in test]
    train_labels = np.array([item["label"] for item in train])
    test_labels = np.array([item["label"] for item in test])

    # Calculate compressed lengths in parallel
    logger.info(
        "Computing compressed lengths for %d train and %d test samples",
        len(train_texts), len(test_texts)
    )
    train_clens = compute_clens_parallel(train_texts)

    logger.info("Computing test text compressed lengths")
    test_clens = compute_clens_parallel(test_texts)

    # Use adaptive strategy to choose the best parallelization approach
    logger.info("Computing distance matrix with adaptive parallel processing")
    dist = compute_distances_adaptive(train_texts, test_texts, train_clens, test_clens)

    # Predict labels using optimized k-NN
    logger.info("Performing k-NN classification")
    predicted_labels = knn_vectorized(dist, train_labels, k=5)

    # Calculate accuracy
    accuracy = np.mean(predicted_labels == test_labels)

    logger.info("=== RESULTS ===")
    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Distance matrix shape: %s", dist.shape)
    logger.info("Cache info: %s", clen.cache_info())
    logger.info("CPU cores used: %d", mp.cpu_count())


if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    mp.set_start_method('spawn', force=True)
    main()
