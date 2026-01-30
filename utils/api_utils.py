# utils/api_utils.py

import json
import os


def token_key_regist(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, value in data.items():
        os.environ[key] = value
        print(f"Registered token for {key}")
