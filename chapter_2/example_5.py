from transformers import pipeline


def classify():
    classifier = pipeline(
        "text-classification",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=0  # remove this line if not using a GPU
    )
    resp = classifier("This restaurant is awesome.")

    return resp


if __name__ == "__main__":
    classification = classify()
    print(classification)
