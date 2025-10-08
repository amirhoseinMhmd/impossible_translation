import spacy
import random

nlp = spacy.load("en_core_web_trf")

REVERSE_MARKER = '🅁'

def noreverse(text: str, seed: int = 42) -> str:
    random.seed(seed)
    doc = nlp(text)
    tokens = [token.text for token in doc]

    if len(tokens) == 0:
        return text

    # Insert 🅁 at a random position
    insert_pos = random.randint(1, len(tokens))
    tokens.insert(insert_pos, REVERSE_MARKER)

    return ' '.join(tokens)


def partialreverse(text: str, seed: int = 42) -> str:
    random.seed(seed)
    doc = nlp(text)
    tokens = [token.text for token in doc]

    if len(tokens) == 0:
        return text

    # Insert 🅁 at a random position
    insert_pos = random.randint(1, len(tokens))

    # Split tokens into before and after R
    before = tokens[:insert_pos]
    after = tokens[insert_pos:]

    # Reverse the tokens after 🅁
    after_reversed = after[::-1]

    result = before + [REVERSE_MARKER] + after_reversed

    return ' '.join(result)


def fullreverse(text: str, seed: int = 42) -> str:
    random.seed(seed)
    doc = nlp(text)
    tokens = [token.text for token in doc]

    if len(tokens) == 0:
        return text

    # Insert 🅁 at a random position
    insert_pos = random.randint(0, len(tokens))
    tokens.insert(insert_pos, REVERSE_MARKER)

    # Reverse all tokens
    tokens_reversed = tokens[::-1]

    return ' '.join(tokens_reversed)


# Example usage
if __name__ == "__main__":
    # Example sentences
    text1 = "He cleans his very messy bookshelf."
    text2 = "They clean his very messy books."
    text3 = "She walks to the store and buys some milk."

    print("=" * 60)
    print("Example 1:")
    print("Original:       ", text1)
    print("NOREVERSE:      ", noreverse(text1))
    print("PARTIALREVERSE: ", partialreverse(text1))
    print("FULLREVERSE:    ", fullreverse(text1))

    print("\n" + "=" * 60)
    print("Example 2:")
    print("Original:       ", text2)
    print("NOREVERSE:      ", noreverse(text2))
    print("PARTIALREVERSE: ", partialreverse(text2))
    print("FULLREVERSE:    ", fullreverse(text2))

    print("\n" + "=" * 60)
    print("Example 3:")
    print("Original:       ", text3)
    print("NOREVERSE:      ", noreverse(text3))
    print("PARTIALREVERSE: ", partialreverse(text3))
    print("FULLREVERSE:    ", fullreverse(text3))

    # Multi-sentence example
    print("\n" + "=" * 60)
    print("Multi-sentence example:")
    text4 = "He walks quickly. She runs faster."
    print("Original:       ", text4)
    print("NOREVERSE:      ", noreverse(text4))
    print("PARTIALREVERSE: ", partialreverse(text4))
    print("FULLREVERSE:    ", fullreverse(text4))

    # Demonstrate different seeds
    print("\n" + "=" * 60)
    print("Different seeds for same text:")
    for s in [10, 20, 30]:
        print(f"\nSeed {s}:")
        print("  NOREVERSE:      ", noreverse(text1, seed=s))
        print("  PARTIALREVERSE: ", partialreverse(text1, seed=s))
        print("  FULLREVERSE:    ", fullreverse(text1, seed=s))