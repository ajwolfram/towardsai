import numpy as np


def compute_perplexity(prob_list):
    np_probs = np.array(prob_list).prod()
    norm_probs = np_probs ** (1 / len(prob_list))
    perplexity = 1 / norm_probs

    return perplexity


if __name__ == "__main__":
    probs = [[0.4, 0.27, 0.55, 0.79], [0.7, 0.5, 0.6, 0.9]]
    for prob_list in probs:
        perplexity = compute_perplexity(prob_list)
        print(f"Perplexity for {prob_list} is {perplexity}")
