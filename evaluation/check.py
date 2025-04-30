from bert_score import score

preds = ["The cat sat on the mat."]
refs = ["The cat is sitting on the mat."]

P, R, F1 = score(preds, refs, lang="en", verbose=True)
print(f"Precision: {P}\nRecall: {R}\nF1: {F1}")