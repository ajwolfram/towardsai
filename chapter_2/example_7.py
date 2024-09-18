import numpy as np


def self_attention(query, key, value, mask=None):
    # compute attention scores
    scores = np.dot(query, key.T)

    if mask is not None:
        # set masked positions to large negative values
        scores = scores + mask * -1e9

    # apply softmax to obtain attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

    # compute weighted sum of value vectors
    output = np.dot(attention_weights, value)

    return output
