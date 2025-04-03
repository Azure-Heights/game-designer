
import os
import re
import json

from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings

from utils import rules_regex


# TODO: create a generated data folder

BASE_DIR = "data/ludii/lud/board"
INDEX_OUTPUT_PATH = "data/game_rules.json"


local_embeddings = OllamaEmbeddings(model="nomic-embed-text")


if __name__ == "__main__":

    game_rules = { }

    for (root, dirs, files) in os.walk(BASE_DIR):
        for f in tqdm(files, desc=root):

            if f.endswith(".lud"):

                with open(os.path.join(root, f), "r") as fin:
                    text = fin.read()

                name = f.split(".")[0]
                rules = rules_regex.search(text).group(1)
                game_rules[name] = {
                    "rules": rules,
                    "embedding": local_embeddings.embed_documents([rules]),
                    "ludeme": text,
                }

    with open(INDEX_OUTPUT_PATH, "w") as fout:
        json.dump(game_rules, fout, indent=4)
