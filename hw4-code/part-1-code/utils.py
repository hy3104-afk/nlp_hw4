import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


random.seed(0)


# Function to introduce typos in words
def introduce_typo(word):
    typo_keys = {
        "a": ["q", "w", "s", "z"],
        "b": ["v", "g", "h", "n"],
        "c": ["x", "d", "f", "v"],
        "d": ["s", "f", "g", "e"],
        "e": ["w", "s", "d", "r"],
        "f": ["r", "g", "d", "s"],
        "g": ["t", "f", "h", "b"],
        "h": ["g", "j", "k", "n"],
        "i": ["u", "o", "k", "l"],
        "j": ["h", "k", "l", "m"],
        "k": ["j", "l", "o", "i"],
        "l": ["k", "o", "p", "j"],
        "m": ["n", "j", "k", "l"],
        "n": ["m", "b", "h", "j"],
        "o": ["i", "k", "p", "u"],
        "p": ["o", "l", "r", "i"],
        "q": ["w", "a", "s", "z"],
        "r": ["t", "f", "e", "d"],
        "s": ["a", "w", "d", "z"],
        "t": ["r", "g", "h", "y"],
        "u": ["i", "o", "y", "p"],
        "v": ["c", "b", "g", "f"],
        "w": ["q", "e", "s", "a"],
        "x": ["z", "c", "v", "d"],
        "y": ["t", "u", "i", "h"],
        "z": ["a", "x", "s", "c"],
    }

    word_list = list(word)
    for i in range(len(word_list)):
        if random.random() < 0.1:  # Introduce typos with 10% probability per letter
            letter = word_list[i]
            if letter in typo_keys:
                word_list[i] = random.choice(typo_keys[letter])
    
    return "".join(word_list)


# Function to replace a word with a synonym using WordNet
def replace_with_synonym(word):
    synsets = wordnet.synsets(word)
    if synsets:
        # Get the first synset
        lemma = synsets[0].lemmas()[0]
        synonym = lemma.name()
        if synonym != word:  # Ensure that we don't replace with the same word
            return synonym
    return word


# Transformation function to apply transformations to the example
def custom_transform(example):
    # Tokenize the text
    words = word_tokenize(example['text'])
    transformed_words = []

    for word in words:
        # Randomly decide to introduce a typo
        if random.random() < 0.1:  # 10% chance to introduce a typo
            word = introduce_typo(word)
        
        # Randomly decide to replace with a synonym
        if random.random() < 0.1:  # 10% chance to replace with a synonym
            word = replace_with_synonym(word)
        
        transformed_words.append(word)

    # Update the example's text with transformed words
    example['text'] = ' '.join(transformed_words)
    return example


# Example transformation to lowercase the text (as given in the initial function)
def example_transform(example):
    example["text"] = example["text"].lower()
    return example