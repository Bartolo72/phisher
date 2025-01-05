# Architektura modelu
[Implementacja](phisher/model/phisher_embeddings_model.py)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .phisher_model import PhisherModel


class PhisherEmbeddingModel(PhisherModel, nn.Module):
    def __init__(
        self: "PhisherEmbeddingModel",
        vocab_size: int,
        embedding_dim: int,
        out_features: int = 1,
    ) -> None:
        PhisherModel.__init__(self, out_features=out_features)
        nn.Module.__init__(self)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 1))
        self.fc1 = nn.Linear(in_features=12 * 47 * 100, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=out_features)

    def forward(self: "PhisherEmbeddingModel", x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))  # Adjust pooling

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))  # Adjust pooling

        x = x.reshape(-1, 12 * 47 * 100)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        return x


```

# Preprocessing danych

Przetwarzanie danych w `PhishingEmbeddingDataset` obejmuje:

1. **Odczyt danych**: Plik CSV z URL-ami i etykietami (`0` - bezpieczny, `1` - phishingowy).
2. **Parsowanie URL-a**: Usunięcie protokołu (`http://`, `https://`) i ścieżki, zachowanie domeny głównej.
   - Przykład: `https://test.pl/index.html` → `test.pl`.
3. **Kodowanie znaków na indeksy**:  
   - Każdy znak mapowany na unikalny indeks (np. `a=1`, `b=2`, ...).  
   - Nieznane znaki są zastępowane przez indeks `0` (padding).
   - Przykładowy alfabet:  
     `{'a': 1, 'b': 2, ..., 'z': 26, 'A': 27, ..., ' ': 84, '0': 53, ..., ';': 83}`.
4. **Normalizacja długości**:  
   - Jeśli domena > 200 znaków: obcinanie.
   - Jeśli < 200 znaków: dopełnienie indeksami `0`.
   - Przykład: `test.pl` → `[46, 5, 19, 20, 84, 16, 12, 0, 0, ...]`.
5. **Zwrot danych**: Każdy przykład zwraca:
   - Tensor indeksów (`200` liczb całkowitych).
   - Etykietę (`0` lub `1`).

**Przykład dla domeny `test.pl`**:  
Wejście: `test.pl`.  
Wyjście:  
- Tensor: `[46, 5, 19, 20, 84, 16, 12, 0, 0, ...]` (długość: `200`).  
- Etykieta: `0` (lub `1`).




# Trenowanie i ewaluacja
Przy uruchamianiu skryptu nalezy zdefiniować adekwatne parametry `model` oraz `dataset`.
```
!python main.py \
    model=phisher_embeddings \
    dataset=embeddings_dataset
```

# Ewaluacja checkpointu
```
!python main.py \
    model=phisher_embeddings \
    dataset=embeddings_dataset \
    model_ckpt=model_path
```
> Gdzie `model_path` nalezy zastąpić ściezką do checkpointu modelu o rozszerzeniu `.ckpt`.