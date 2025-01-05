# Architektura modelu
[Implementacja](phisher/model/phisher_one_hot_model.py)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .phisher_model import PhisherModel


class PhisherOneHotModel(PhisherModel, nn.Module):
    def __init__(self: "PhisherOneHotModel", out_features: int = 1) -> None:
        PhisherModel.__init__(self, out_features=out_features)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5))

        self.fc1 = nn.Linear(in_features=12 * 47 * 18, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=out_features)

    def forward(self: "PhisherOneHotModel", x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten
        x = x.reshape(-1, 12 * 47 * 18)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        return x

```

# Preprocessing danych

Przetwarzanie danych w `PhishingOneHotDataset` obejmuje:

1. **Odczyt danych**: Plik CSV z URL-ami i etykietami (`0` - bezpieczny, `1` - phishingowy).
2. **Parsowanie URL-a**: Usunięcie protokołu (`http://`, `https://`) i ścieżki, zachowanie domeny głównej.
   - Przykład: `https://test.pl/index.html` → `test.pl`.
3. **Normalizacja długości**:  
   - Jeśli domena > 200 znaków: obcinanie.
   - Jeśli < 200 znaków: dopełnienie spacjami.
   - Przykład: `test.pl` → `test.pl␣␣␣␣...`.
4. **Kodowanie One-Hot**:  
   - Każdy znak zamieniany jest na macierz one-hot o rozmiarze `200 x 84`.
   - Alfabet obejmuje litery, cyfry, znaki specjalne oraz spacje.
   - Przykład (domena `test.pl`):  

| **t** | **e** | **s** | **t** | **.** | **p** | **l** | **␣** | ... |
|-------|-------|-------|-------|-------|-------|-------|-------|-----|
| 0     | 0     | 0     | 1     | 0     | 0     | 0     | 0     | ... |
| 0     | 1     | 0     | 0     | 0     | 0     | 0     | 0     | ... |
| 0     | 0     | 1     | 0     | 0     | 0     | 0     | 0     | ... |



# Trenowanie i ewaluacja
Przy uruchamianiu skryptu nalezy zdefiniować adekwatne parametry `model` oraz `dataset`.
```
!python /content/phisher/main.py \
    model=phisher_one_hot \
    dataset=one_hot_dataset
```

# Ewaluacja checkpointu
```
!python /content/phisher/main.py \
    model=phisher_one_hot \
    dataset=one_hot_dataset \
    model_ckpt=model_path
```
> Gdzie `model_path` nalezy zastąpić ściezką do checkpointu modelu o rozszerzeniu `.ckpt`.