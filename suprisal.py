import random

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import math

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def token_surprisal(text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits[:, :-1, :]
        probs = torch.softmax(logits, dim=-1)

    surprisals = []
    for i in range(input_ids.size(1) - 1):
        token_id = input_ids[0, i+1]
        prob = probs[0, i, token_id].item()
        surprisals.append(-math.log2(prob))

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return list(zip(tokens[1:], surprisals))

def avg_surprisal(text):
    surprisals = [s for _, s in token_surprisal(text)]
    return sum(surprisals) / len(surprisals)

def perplexity(text):
    return 2 ** avg_surprisal(text)


text_original = "The quick brown fox jumps over the lazy dog"


print("Token surprisals:", token_surprisal(text_original))
print("Average surprisal:", avg_surprisal(text_original), "bits/token")
print("Perplexity:", perplexity(text_original))

print("-" * 60)

words = text_original.split()
random.shuffle(words)
shuffled_text = ' '.join(words)

print("Token surprisals:", token_surprisal(shuffled_text))
print("Average surprisal:", avg_surprisal(shuffled_text), "bits/token")
print("Perplexity:", perplexity(shuffled_text))