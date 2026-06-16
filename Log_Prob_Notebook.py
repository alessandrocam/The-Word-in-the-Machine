# NOTE:
# Meta's Llama-3.2-1B is a gated model.
# To use it, you must first review and accept the Llama 3.2 Community License Agreement
# on Hugging Face, and obtain an access token:
# https://huggingface.co/meta-llama/Llama-3.2-1B
#
# Then store your token in your Colab Secrets before running this notebook.

# 1. Install necessary libraries
!pip install -q transformers accelerate huggingface_hub

# 2. Imports
import os
import torch

from google.colab import userdata
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = userdata.get("HF_TOKEN") # Retrieves Hugging Face access token from Colab Secrets (the name for the token is here set as "HF_TOKEN")

if HF_TOKEN:
    login(HF_TOKEN)
else:
    raise ValueError(
        "HF_TOKEN not found in Colab userdata. Please add your Hugging Face token."
    )

# 4. Load model and tokenizer
model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto"
)

model.eval()

# Optional: confirm GPU
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# 5. Function to compute log P(B | A)
def log_prob(A, B):

    device = model.device

    # Tokenize prompt alone
    tokens_A = tokenizer(A, return_tensors="pt").to(device)

    # Tokenize prompt + continuation
    tokens_full = tokenizer(A + B, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**tokens_full)

    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    input_ids = tokens_full["input_ids"][0]

    len_A = tokens_A["input_ids"].shape[1]

    total_log_prob = 0.0

    # Sum token log probabilities for B only
    for i in range(len_A, len(input_ids)):
        token_id = input_ids[i]
        total_log_prob += log_probs[0, i - 1, token_id].item()

    return total_log_prob

# 6. Example experiment
A = "The capital of France is"

B1 = " Paris."
B2 = " London."

score1 = log_prob(A, B1)
score2 = log_prob(A, B2)

print()
print("log P(Paris | A)  =", score1)
print("log P(London | A) =", score2)

def log_prob(A, B):
    device = model.device

    # Standardize the transition
    # We ensure B starts with a space if it doesn't have one,
    # reflecting natural linguistic flow after a period.
    if not B.startswith(" "):
        B = " " + B

    # Tokenize the full sequence ONLY
    # This ensures we see how A and B interact/merge.
    full_text = A + B
    tokens_full = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = tokens_full["input_ids"][0]

    # Robust split-point detection
    # We tokenize A alone just to find its length in the context of the full string.
    # We use 'add_special_tokens=True' to ensure the BOS token is accounted for.
    tokens_A = tokenizer(A, return_tensors="pt")
    len_A = tokens_A["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(tokens_full["input_ids"])

    # Proper Logit Shifting
    # Logits at index i predict the token at index i+1.
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    total_log_prob = 0.0

    # Cleaner indexing loop
    # We start from len_A because that is the first token of "B".
    # We pull the probability from the logit at (i - 1).
    for i in range(len_A, len(input_ids)):
        token_id = input_ids[i]
        # Get the model's confidence for this token based on all previous tokens
        token_score = log_probs[0, i - 1, token_id].item()
        total_log_prob += token_score

    return total_log_prob

# Example experiment
A = "I have just stepped on a nail. Ouch,"
B = "it hurts!"

score = log_prob(A, B)

print(f"\nlog P(B | A) = {score:.4f}")
