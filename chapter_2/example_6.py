from transformers import AutoModel, pipeline


def finish_sentence(inp):
    gpt2 = AutoModel.from_pretrained("gpt2")
    print(gpt2)

    generator = pipeline(
        model="gpt2",
        device=0  # remove this line if not using a GPU
    )
    output = generator(
        inp,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=4,
        max_new_tokens=50,
        return_full_text=False,
    )

    for item in output:
        print(">", item["generated_text"], "\n")


if __name__ == "__main__":
    finish_sentence("This movie was a very")
