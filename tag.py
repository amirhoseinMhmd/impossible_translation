import sys
sys.path.append("..")

import tqdm
import os
import argparse
import json
import spacy
import torch
from sentence_transformers import SentenceTransformer
import benepar

def __get_constituency_parse(sent):
    try:
        tree = sent._.parse_string
        return f"(ROOT {tree})"
    except Exception:
        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Tag BabyLM dataset',
        description='Tag BabyLM dataset using spaCy + Sentence-Transformers + Benepar')
    parser.add_argument('path', type=argparse.FileType('r'),
                        nargs='+', help="Path to file(s)")
    parser.add_argument('-p', '--parse', action='store_true',
                        help="Include constituency parse")
    args = parser.parse_args()

    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load spaCy with Benepar
    nlp_spacy = spacy.load("en_core_web_trf", disable=["ner"])
    if args.parse:
        if not benepar.is_loaded('benepar_en3'):
            benepar.download('benepar_en3')
        nlp_spacy.add_pipe("benepar", config={"model": "benepar_en3"})

    # Load Sentence-Transformers
    model = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))

    BATCH_SIZE = 1000

    for file in args.path:
        print(file.name)
        lines = [l.strip() for l in file.readlines()]
        line_batches = [lines[i:i + BATCH_SIZE] for i in range(0, len(lines), BATCH_SIZE)]
        text_batches = [" ".join(l) for l in line_batches]

        line_annotations = []
        print("Segmenting, annotating, and embedding...")

        for text in tqdm.tqdm(text_batches):
            doc = nlp_spacy(text)

            sent_annotations = []
            for sent in doc.sents:
                word_annotations = []
                for i, token in enumerate(sent, start=1):
                    feats_str = "|".join(f"{k}={v}" for k, v in token.morph.to_dict().items()) or None
                    wa = {
                        'id': i,
                        'text': token.text,
                        'lemma': token.lemma_,
                        'upos': token.pos_,
                        'xpos': token.tag_,
                        'feats': feats_str,
                        'start_char': token.idx,
                        'end_char': token.idx + len(token.text)
                    }
                    word_annotations.append(wa)

                emb = model.encode(sent.text, convert_to_numpy=True).tolist()

                sa = {
                    'sent_text': sent.text,
                    'word_annotations': word_annotations,
                    'embedding': emb
                }

                if args.parse:
                    sa['constituency_parse'] = __get_constituency_parse(sent)

                sent_annotations.append(sa)

            line_annotations.append({'sent_annotations': sent_annotations})

        ext = '_parsed.json' if args.parse else '.json'
        out_name = os.path.splitext(file.name)[0] + ext
        with open(out_name, "w") as outfile:
            json.dump(line_annotations, outfile, indent=4)

        print(f"Saved: {out_name}")