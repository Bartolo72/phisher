import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
import pandas as pd


ALPHABET = list(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;"
) + [" "]


def encode_url(url: str, max_length: int) -> torch.Tensor:
    char_to_idx = {char: idx for idx, char in enumerate(ALPHABET, start=1)}
    pad_idx = 0

    parsed_url = url.split("://")[-1]
    if "/" in parsed_url:
        parsed_url = parsed_url.split("/")[0]

    url_indices = [char_to_idx.get(char, pad_idx) for char in url]
    if len(url_indices) > max_length:
        url_indices = url_indices[:max_length]
    else:
        url_indices.extend([pad_idx] * (max_length - len(url_indices)))
    return torch.tensor(url_indices, dtype=torch.long)


class PhishingEmbeddingDataset(Dataset):
    def __init__(
        self: "PhishingEmbeddingDataset", csv_file_path: str, max_length: int = 200
    ) -> None:
        self.data: pd.DataFrame = pd.read_csv(csv_file_path)
        self.max_length: int = max_length

    def __len__(self: "PhishingEmbeddingDataset") -> int:
        return len(self.data)

    def __getitem__(
        self: "PhishingEmbeddingDataset", idx: int
    ) -> Tuple[torch.Tensor, int]:
        url = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        return encode_url(url, self.max_length), label

    def get_stats(self: "PhishingEmbeddingDataset") -> Dict[str, Any]:
        labels = self.data["label"].unique()
        stats = [
            {
                "Label": label,
                "Count": len(self.data[self.data["label"] == label]),
                "Percent": round(
                    len(self.data[self.data["label"] == label]) / len(self.data) * 100,
                    2,
                ),
            }
            for label in labels
        ]
        return stats

    def print_stats(self: "PhishingEmbeddingDataset") -> None:
        stats = self.get_stats()
        for stat in stats:
            print(
                f"Label: {stat['Label']}, Count: {stat['Count']}, Percent: {stat['Percent']}%"
            )
