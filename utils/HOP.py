import spacy

nlp = spacy.load("en_core_web_trf")
SINGULAR_MARKER = '🅂'
PLURAL_MARKER = '🄿'

def nohop(text: str) -> str:
    doc = nlp(text)
    result = []

    for token in doc:
        # Check if it's a 3rd person present tense verb
        if is_3rd_person_present_verb(token):
            # Use spaCy's lemma
            result.append(token.lemma_)
            result.append(' ')  # Space before marker
            # Add marker immediately after
            marker = SINGULAR_MARKER if is_singular_verb(token) else PLURAL_MARKER
            result.append(marker)
        else:
            result.append(token.text)

        # Preserve spacing
        if token.whitespace_:
            result.append(' ')

    return ''.join(result).strip()


def tokenhop(text: str) -> str:
    doc = nlp(text)
    tokens = list(doc)
    result = []
    pending_markers = {}  # {insert_index: marker}

    for i, token in enumerate(tokens):
        # Add any pending marker at this position
        if i in pending_markers:
            result.append(pending_markers[i])
            result.append(' ')

        if is_3rd_person_present_verb(token):
            # Use spaCy's lemma
            result.append(token.lemma_)
            # Schedule marker to be inserted 4 tokens later
            marker = SINGULAR_MARKER if is_singular_verb(token) else PLURAL_MARKER
            insert_index = i + 4
            # Store the marker to insert later (handle if multiple markers at same position)
            if insert_index in pending_markers:
                pending_markers[insert_index] += ' ' + marker
            else:
                pending_markers[insert_index] = marker
        else:
            result.append(token.text)

        # Preserve spacing
        if token.whitespace_:
            result.append(' ')

    # Add any remaining markers at the end
    for idx in sorted(pending_markers.keys()):
        if idx >= len(tokens):
            result.append(' ')
            result.append(pending_markers[idx])

    return ''.join(result).strip()


def wordhop(text: str) -> str:
    doc = nlp(text)
    tokens = list(doc)
    result = []
    pending_markers = []  # List of (target_word_count, marker) tuples
    word_count = 0

    for i, token in enumerate(tokens):
        # Check if we should insert any pending markers at this word count
        markers_to_insert = [m for wc, m in pending_markers if wc == word_count]
        for marker in markers_to_insert:
            result.append(marker)
            result.append(' ')
        # Remove inserted markers
        pending_markers = [(wc, m) for wc, m in pending_markers if wc != word_count]

        if is_3rd_person_present_verb(token):
            # Use spaCy's lemma
            result.append(token.lemma_)
            # Schedule marker to be inserted 4 words after this verb
            marker = SINGULAR_MARKER if is_singular_verb(token) else PLURAL_MARKER
            target_wc = word_count + 4 + (1 if not token.is_punct else 0)
            pending_markers.append((target_wc, ' ' + marker))
        else:
            result.append(token.text)

        # Increment word count (skip punctuation)
        if not token.is_punct:
            word_count += 1

        # Preserve spacing
        if token.whitespace_:
            result.append(' ')

    # Add any remaining markers at the end
    for _, marker in pending_markers:
        result.append(' ')
        result.append(marker)

    return ''.join(result).strip().replace('  ', ' ')


def is_3rd_person_present_verb(token) -> bool:
    # Check for present tense verbs
    if token.tag_ in ['VBZ', 'VBP']:
        return True
    # Also check using morphological features if available
    if token.pos_ == 'VERB':
        morph = token.morph.to_dict()
        if morph.get('Tense') == 'Pres' and morph.get('VerbForm') == 'Fin':
            return True
    return False


def is_singular_verb(token) -> bool:
    if token.tag_ == 'VBZ':
        return True
    # Check morphological features
    morph = token.morph.to_dict()
    if morph.get('Number') == 'Sing' and morph.get('Person') == '3':
        return True
    return False


if __name__ == "__main__":
    # Example sentences
    text1 = "He cleans his very messy bookshelf."
    text2 = "They clean their very messy bookshelf."
    text3 = "She walks to the store and buys some milk."

    print("=" * 60)
    print("Example 1:")
    print("Original: ", text1)
    print("NOHOP:    ", nohop(text1))
    print("TOKENHOP: ", tokenhop(text1))
    print("WORDHOP:  ", wordhop(text1))

    print("\n" + "=" * 60)
    print("Example 2:")
    print("Original: ", text2)
    print("NOHOP:    ", nohop(text2))
    print("TOKENHOP: ", tokenhop(text2))
    print("WORDHOP:  ", wordhop(text2))

    print("\n" + "=" * 60)
    print("Example 3:")
    print("Original: ", text3)
    print("NOHOP:    ", nohop(text3))
    print("TOKENHOP: ", tokenhop(text3))
    print("WORDHOP:  ", wordhop(text3))

    # Multi-sentence example
    print("\n" + "=" * 60)
    print("Multi-sentence example:")
    text4 = "He walks quickly. She runs faster. They play together."
    print("Original: ", text4)
    print("NOHOP:    ", nohop(text4))
    print("TOKENHOP: ", tokenhop(text4))
    print("WORDHOP:  ", wordhop(text4))