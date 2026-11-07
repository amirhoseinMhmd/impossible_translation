"""Recover the exact GPT-2 token permutation behind each perturbed input."""
import itertools
import random

from numpy.random import default_rng
from transformers import GPT2TokenizerFast

tok = GPT2TokenizerFast.from_pretrained("gpt2")
REVERSE_MARKER = '\U0001f141'


def _pieces(ids):
    return [tok.decode([i]) for i in ids]


def encode_with_offsets(text):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    return enc["input_ids"], enc["offset_mapping"]


def recover_full_shuffle(actual, inp):
    text = actual.strip()
    ids, offs = encode_with_offsets(text)
    order = list(range(len(ids)))
    default_rng(57).shuffle(order)
    ok = tok.decode([ids[i] for i in order]).strip() == inp.strip()
    pos = {orig: new for new, orig in enumerate(order)}
    return text, ids, offs, pos, ok


def recover_partial_reverse(actual, inp):
    text = actual.strip()
    ids, offs = encode_with_offsets(text)
    n = len(ids)
    if n < 3:
        return text, ids, offs, None, False
    random.seed(42)
    k = random.randint(0, n - 2)
    order = list(range(k)) + [None] + list(range(n - 1, k - 1, -1))
    pieces = _pieces(ids)
    rebuilt = ' '.join([pieces[i] if i is not None else REVERSE_MARKER for i in order]).replace('  ', ' ')
    ok = rebuilt.strip() == inp.strip()
    pos = {orig: new for new, orig in enumerate(order) if orig is not None}
    return text, ids, offs, pos, ok


def recover_local_shuffle(actual, inp, window=3):
    text = ' ' + actual + ' '
    ids, offs = encode_with_offsets(text)
    target = inp.strip()
    windows = [list(range(i, min(i + window, len(ids)))) for i in range(0, len(ids), window)]
    order = []

    def dfs(w):
        if w == len(windows):
            return tok.decode([ids[i] for i in order]).strip() == target
        for perm in set(itertools.permutations(windows[w])):
            order.extend(perm)
            # trailing partial UTF-8 bytes decode to U+FFFD, so drop them before the prefix test
            prefix = tok.decode([ids[i] for i in order]).lstrip().rstrip('�').rstrip()
            if target.startswith(prefix) and dfs(w + 1):
                return True
            del order[-len(perm):]
        return False

    ok = dfs(0)
    pos = {orig: new for new, orig in enumerate(order)} if ok else None
    return text, ids, offs, pos, ok


RECOVER = {
    'fullShuffle': recover_full_shuffle,
    'partialReverse': recover_partial_reverse,
    'localShuffle3': recover_local_shuffle,
}


def word_to_first_piece(doc_tokens, offs, char_shift):
    """Map each spaCy token (idx, len) to the index of its first overlapping GPT-2 piece."""
    out = []
    for start, length in doc_tokens:
        s, e = start + char_shift, start + char_shift + length
        first = None
        for j, (a, b) in enumerate(offs):
            if b > s and a < e:
                first = j
                break
        out.append(first)
    return out
