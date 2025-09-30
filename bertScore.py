from bert_score import score

# Same words, different order
# text1 = ["The quick brown fox jumps over the lazy dog"]
# text2 = ["The lazy dog jumps over the quick brown fox"]
text1 = ["future The of AI is now."]
text2 = ["The future is of now AI."]
P, R, F1 = score(text1, text2, lang="en", model_type="bert-base-uncased")

print(f"Precision: {P.item():.4f}")
print(f"Recall: {R.item():.4f}")
print(f"F1: {F1.item():.4f}")