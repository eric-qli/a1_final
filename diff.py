import torch
import vocab

from q2_algorithm import mstSingleRoot

transtion_model = torch.load("weights-q1.pt")
graph_model = torch.load("weights-q2.pt")


tokens = ["<ROOT>", "To", "raise", "those", "doubts", "is", "to", "resolve", "them", "."]
word_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
word_tensor = torch.tensor([word_ids])
lengths = torch.tensor([len(tokens)-1]) 

# graph
arc_scores, label_scores = graph_model(word_tensor, lengths)
heads = mstSingleRoot(arc_scores, lengths)
labels = label_scores.argmax(-1)

#transiton
heads, labels = transtion_model.parse(word_tensor, lengths)

for i, tok in enumerate(tokens):
    head = tokens[heads[i]] if heads[i] != 0 else "ROOT"
    print(f"{tok:10s} <-- {head:10s} ({labels[i]})")
