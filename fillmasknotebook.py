# -*- coding: utf-8 -*-
"""FillMaskNotebook.ipynb

"""

# Install the transformers library
!pip install transformers

# Import the necessary libraries
from transformers import pipeline, RobertaTokenizer, RobertaForMaskedLM
import torch

# Load the pre-trained RoBERTa model and tokenizer
model_name = 'roberta-large'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)

# Create a fill-mask pipeline
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

# The sentence with the <mask> token where the blank is
sentence = "The rose <mask> red."

# Get the top 10 predictions for the masked token
predictions = fill_mask(sentence, top_k=10)

# Print the top 10 predicted tokens with their probabilities
for prediction in predictions:
    token_str = prediction['token_str']
    score = prediction['score']
    print(f"{token_str}: {score:.4f}")

