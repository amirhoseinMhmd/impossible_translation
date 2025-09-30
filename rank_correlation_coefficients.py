from scipy.stats import kendalltau, spearmanr
import random

# --- Setup ---
text_original = "The quick brown fox jumps over the lazy dog"
tokens_original = text_original.split()

# Create the original sequence of indices [0, 1, 2, ..., 8]
original_indices = list(range(len(tokens_original)))

# --- Version 1: Slightly Shuffled Text ---
tokens_slightly_shuffled = tokens_original[:]
# Swap just two adjacent words
tokens_slightly_shuffled[1], tokens_slightly_shuffled[2] = tokens_slightly_shuffled[2], tokens_slightly_shuffled[1]
text_slightly_shuffled = " ".join(tokens_slightly_shuffled)
# Get the new sequence of indices
shuffled_indices_1 = [tokens_original.index(token) for token in tokens_slightly_shuffled]

# --- Version 2: Heavily Shuffled Text ---
tokens_heavily_shuffled = tokens_original[:]
random.shuffle(tokens_heavily_shuffled)
text_heavily_shuffled = " ".join(tokens_heavily_shuffled)
# Get the new sequence of indices
shuffled_indices_2 = [tokens_original.index(token) for token in tokens_heavily_shuffled]


# --- Evaluation ---
# Compare the original index order with the shuffled index orders
kendall_slightly, _ = kendalltau(original_indices, shuffled_indices_1)
spearman_slightly, _ = spearmanr(original_indices, shuffled_indices_1)

kendall_heavily, _ = kendalltau(original_indices, shuffled_indices_2)
spearman_heavily, _ = spearmanr(original_indices, shuffled_indices_2)

# --- Print Results ---
print(f"Original Text:    '{text_original}'\n")

print(f"Slightly Shuffled: '{text_slightly_shuffled}'")
print(f"  Kendall's Tau: {kendall_slightly:.4f}")
print(f"  Spearman's Rho: {spearman_slightly:.4f}\n")

print(f"Heavily Shuffled:  '{text_heavily_shuffled}'")
print(f"  Kendall's Tau: {kendall_heavily:.4f}")
print(f"  Spearman's Rho: {spearman_heavily:.4f}")