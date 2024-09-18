from transformers import AutoModel

BART = AutoModel.from_pretrained("facebook/bart-large")
print(BART)
