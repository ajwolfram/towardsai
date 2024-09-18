import os
import shutil

from dotenv import load_dotenv

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_IDS = [
    # "meta-llama/Llama-2-7b-chat-hf",
    # "tiiuae/falcon-7b-instruct",
    "databricks/dolly-v2-3b",
]


def translate_english_to_french(prompt, model_id):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    offload_dir = f"{current_dir}/offload_dir"
    os.makedirs(offload_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # will utilize a GPU if you have one, which speeds things up
        device_map="auto",
        # only needed for the falcon-7b-instruct model when setting device_map
        offload_folder=offload_dir,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        # prevents a warning message in the logs
        clean_up_tokenization_spaces=True,
    )

    # when using a GPU and device_map="auto" above, we need to put tensors in a single location
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
        "cuda"
    )
    outputs = model.generate(**inputs, max_new_tokens=100)

    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    shutil.rmtree(offload_dir)

    return decoded_output


def run_demo():
    prompt = "Translate the following English phrase into French: Configuration files are easy to use!"
    for model_id in MODEL_IDS:
        french = translate_english_to_french(prompt, model_id)
        print(f"{model_id}: {french}")


if __name__ == "__main__":
    run_demo()
