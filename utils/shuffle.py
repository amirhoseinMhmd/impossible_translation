from transformers import GPT2Tokenizer
from numpy.random import default_rng
from multiprocessing import Pool, cpu_count
from typing import List
from tqdm import tqdm

tokenizer = None


def _init_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token


_init_tokenizer()


def noshuffle(text: str) -> str:
    text = text.strip()
    return text


def local_shuffle(text: str, window_size: int = 3, seed: int = None) -> str:
    text = ' ' + text + ' '

    # Get token IDs (like in the paper)
    tokens = tokenizer.encode(text)

    if len(tokens) < window_size:
        return text

    # Shuffle tokens in batches of size window
    shuffled_tokens = []
    for i in range(0, len(tokens), window_size):
        batch = tokens[i:i + window_size].copy()

        # Use numpy RNG (same as paper)
        if seed is not None:
            default_rng(seed).shuffle(batch)
        else:
            default_rng().shuffle(batch)

        shuffled_tokens += batch

    # Decode back to text
    return tokenizer.decode(shuffled_tokens).strip()


def full_shuffle(text: str, seed: int = None) -> str:
    text = text.strip()

    # Get token IDs
    tokens = tokenizer.encode(text)

    if len(tokens) == 0:
        return text

    # Shuffle all tokens
    shuffled_tokens = tokens.copy()
    if seed is not None:
        default_rng(seed).shuffle(shuffled_tokens)
    else:
        default_rng().shuffle(shuffled_tokens)

    # Decode back to text
    return tokenizer.decode(shuffled_tokens)


def _process_chunk_local_shuffle_batched(args):
    chunk, batch_size, window_size = args
    results = []
    for i in range(0, len(chunk), batch_size):
        batch = chunk[i:i + batch_size]
        results.extend(_local_shuffle_batch(batch, window_size))
    return results


def _chunk_list(lst: List, n_chunks: int) -> List[List]:
    chunk_size = len(lst) // n_chunks + (1 if len(lst) % n_chunks else 0)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def _local_shuffle_batch(texts: List[str], window_size: int = 3) -> List[str]:
    texts = [t.strip() for t in texts]
    results = []

    for text in texts:
        # Add space padding (same as local_shuffle)
        text = ' ' + text + ' '

        # Get token IDs
        tokens = tokenizer.encode(text)

        if len(tokens) < window_size:
            results.append(text.strip())  # Strip before appending
            continue

        # Shuffle tokens in batches of size window
        shuffled_tokens = []
        for i in range(0, len(tokens), window_size):
            batch = tokens[i:i + window_size].copy()
            default_rng().shuffle(batch)
            shuffled_tokens += batch

        # Decode back to text and strip (same as local_shuffle)
        results.append(tokenizer.decode(shuffled_tokens).strip())

    return results

def _full_shuffle_batch(texts: List[str]) -> List[str]:
    texts = [t.strip() for t in texts]
    results = []

    for text in texts:
        # Get token IDs
        tokens = tokenizer.encode(text)

        if len(tokens) == 0:
            results.append(text)
            continue

        # Shuffle all tokens
        shuffled_tokens = tokens.copy()
        default_rng().shuffle(shuffled_tokens)

        # Decode back to text
        results.append(tokenizer.decode(shuffled_tokens))

    return results


def local_shuffle_fast(texts: List[str], window_size: int = 3, batch_size: int = 128, n_workers: int = None) -> List[
    str]:
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    if len(texts) < 500:
        return _local_shuffle_batch(texts, window_size)

    chunks = _chunk_list(texts, n_workers)
    chunk_args = [(chunk, batch_size, window_size) for chunk in chunks]

    with Pool(n_workers) as pool:
        results_nested = pool.map(_process_chunk_local_shuffle_batched, chunk_args)

    results = []
    for chunk_results in results_nested:
        results.extend(chunk_results)

    return results


def full_shuffle_fast(texts: List[str], batch_size: int = 128, n_workers: int = None) -> List[str]:
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    if len(texts) < 500:
        return _full_shuffle_batch(texts)

    chunks = _chunk_list(texts, n_workers)
    chunk_args = [(chunk, batch_size) for chunk in chunks]

    results = []
    with Pool(n_workers) as pool:
        for chunk in chunks:
            chunk_results = _full_shuffle_batch(chunk)
            results.extend(chunk_results)

    return results


def local_shuffle_batch(texts: List[str], batch_size: int = 128, window_size: int = 3,  n_workers: int = None) -> List[
    tuple]:
    results = []

    for text in tqdm(texts):
        results.append((local_shuffle(text, window_size), text))

    return results


def full_shuffle_batch(texts: List[str], batch_size: int = 128, n_workers: int = None) -> List[tuple]:

    originals = [t.strip() for t in texts]
    shuffled = full_shuffle_fast(texts, batch_size, n_workers)
    return list(zip(shuffled, originals))


# Example usage
if __name__ == "__main__":
    # Example sentences
    text1 = "He cleans his very messy bookshelf."
    text2 = "They clean his very messy books."
    text3 = "No it's not bloody December."

    print("=" * 60)
    print("Example 1:")
    print("Original:           ", text1)
    print("NOSHUFFLE:          ", noshuffle(text1))
    print("LOCAL_SHUFFLE (w=3):", local_shuffle(text1, window_size=3, seed=42))
    print("LOCAL_SHUFFLE (w=5):", local_shuffle(text1, window_size=5, seed=42))
    print("FULL_SHUFFLE:       ", full_shuffle(text1, seed=42))

    print("\n" + "=" * 60)
    print("Example 2:")
    print("Original:           ", text2)
    print("NOSHUFFLE:          ", noshuffle(text2))
    print("LOCAL_SHUFFLE (w=3):", local_shuffle(text2, window_size=3, seed=42))
    print("LOCAL_SHUFFLE (w=5):", local_shuffle(text2, window_size=5, seed=42))
    print("FULL_SHUFFLE:       ", full_shuffle(text2, seed=42))

    print("\n" + "=" * 60)
    print("Example 3:")
    print("Original:           ", text3)
    print("NOSHUFFLE:          ", noshuffle(text3))
    print("LOCAL_SHUFFLE (w=3):", local_shuffle(text3, window_size=3, seed=42))
    print("LOCAL_SHUFFLE (w=5):", local_shuffle(text3, window_size=5, seed=42))
    print("FULL_SHUFFLE:       ", full_shuffle(text3, seed=42))
