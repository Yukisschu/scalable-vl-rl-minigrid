import os
import json
import numpy
import re
import torch
import torch_ac
import gymnasium as gym

# The function is referenced from the official Minigrid starter repo to match the style, with some modifications. 
# @inproceedings{MinigridMiniworld23,
#   author       = {Maxime Chevalier{-}Boisvert and Bolun Dai and Mark Towers and Rodrigo Perez{-}Vicente and Lucas Willems and Salem Lahlou and Suman Pal and Pablo Samuel Castro and Jordan Terry},
#   title        = {Minigrid {\&} Miniworld: Modular {\&} Customizable Reinforcement Learning Environments for Goal-Oriented Tasks},
#   booktitle    = {Advances in Neural Information Processing Systems 36, New Orleans, LA, USA},
#   month        = {December},
#   year         = {2023},
# }
def get_obss_preprocessor(obs_space):
    # Check if obs_space is a plain image space (e.g. pixel-only environments)
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space (image + text mission)
    elif isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces.keys():
        # Fix the text capacity at 100 tokens; MiniGrid missions are short
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        # Build a shared vocabulary that grows lazily as new tokens are seen
        vocab = Vocabulary(obs_space["text"])

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            })

        # Attach the vocabulary to the closure so callers can inspect it
        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Converting to a numpy array first avoids a known PyTorch performance issue
    # where creating a tensor directly from a list of arrays is very slow.
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        # Extract only lowercase alphabetic tokens, ignoring punctuation and digits
        tokens = re.findall("([a-z]+)", text.lower())
        # Map each token to its vocabulary index, adding new tokens lazily
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        # Track the longest sequence to determine the padding width
        max_text_len = max(len(var_indexed_text), max_text_len)

    # Allocate a zero-padded matrix; index 0 serves as the implicit padding value
    indexed_texts = numpy.zeros((len(texts), max_text_len))

    # Copy each variable-length sequence into the fixed-width matrix (left-aligned)
    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        # Internal dict mapping token strings to integer indices (1-based)
        self.vocab = {}

    def load_vocab(self, vocab):
        # Replace the internal vocabulary with a previously saved dict
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            # Assign the next integer index; indices are 1-based so that 0
            # remains available as an implicit padding / unknown-token value.
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
