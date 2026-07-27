import os
import shutil

from huggingface_hub import hf_hub_download

import config

REPO_ID = "Xenova/all-MiniLM-L6-v2"
FILES = [
    ("tokenizer.json", "tokenizer.json"),
    ("onnx/model.onnx", "model.onnx"),
]


def download() -> None:
    target_dir = config.EMBEDDING_MODEL_PATH
    os.makedirs(target_dir, exist_ok=True)

    for repo_path, local_name in FILES:
        downloaded = hf_hub_download(repo_id=REPO_ID, filename=repo_path)
        shutil.copyfile(downloaded, os.path.join(target_dir, local_name))
        print(f"{repo_path} -> {os.path.join(target_dir, local_name)}")


if __name__ == "__main__":
    download()
