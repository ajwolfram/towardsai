from transformers import AutoModel

BERT = AutoModel.from_pretrained("bert-base-uncased")
print(BERT)
