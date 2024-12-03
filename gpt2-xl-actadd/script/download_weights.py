import shutil
import os
from transformer_lens import HookedTransformer

MODEL = "gpt2-xl"
CACHE_DIR = "weights"

if os.path.exists(CACHE_DIR):
    if "y" in input(f"{CACHE_DIR} exists, delete and redownload? (y/n)"):
        shutil.rmtree(CACHE_DIR)
    else:
        print("abort; files exist")
        exit()

os.makedirs(CACHE_DIR)

model = HookedTransformer.from_pretrained(MODEL, cache_dir=CACHE_DIR)

print(model)
