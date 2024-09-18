from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def tokenize(inp, tokenizer):
    return tokenizer(inp, return_tensors="pt")


def embed_tokens(inp_tokenized, model):
    return model.model.decoder.embed_tokens(inp_tokenized["input_ids"])


def embed_positions(inp_tokenized, model):
    return model.model.decoder.embed_positions(inp_tokenized["attention_mask"])


def self_attn(inp_attention, model):
    hidden_states, _, _ = model.model.decoder.layers[0].self_attn(inp_attention)
    return hidden_states


def run_demo(inp):
    """
        The "low_cpu_mem_usage arg below automatically gets set to True when
        we set a quantization, so it's unnecessary, but setting it explicitly
        prevents an addtional log message which I think is cleaner.
    """
    OPT = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/opt-1.3b",
        # prevents a warning message in the logs
        clean_up_tokenization_spaces=True
    )
    inp_tokenized = tokenize(inp, tokenizer)
    print(inp_tokenized["input_ids"].size())
    print(inp_tokenized)

    print(OPT.model)

    inp_embedded = embed_tokens(inp_tokenized, OPT)
    print("Layer:\t", OPT.model.decoder.embed_tokens)
    print("Size:\t", inp_embedded.size())
    print("Output:\t", inp_embedded)

    inp_pos_embedded = embed_positions(inp_tokenized, OPT)
    print("Layer:\t", OPT.model.decoder.embed_positions)
    print("Size:\t", inp_pos_embedded.size())
    print("Output:\t", inp_pos_embedded)

    inp_attention = inp_embedded + inp_pos_embedded
    hidden_states = self_attn(inp_attention, OPT)
    print("Layer:\t", OPT.model.decoder.layers[0].self_attn)
    print("Size:\t", hidden_states.size())
    print("Output:\t", hidden_states)


if __name__ == "__main__":
    run_demo("The quick brown fox jumps over the lazy dog")
