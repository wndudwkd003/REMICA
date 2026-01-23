# utils/data_utils.py


import json

from torch.utils.data import Dataset


class JsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, meta_to_text: bool = False):
        self.data = []
        self.meta_to_text = meta_to_text

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sid = sample["id"]
        text = sample["text"]
        label = sample["label"]
        metadata = sample["metadata"]
        if self.meta_to_text:
            text = f"{text} {metadata}"
        return str(sid), str(text), int(label), metadata
