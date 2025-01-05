# Architektura modelu
![Architektura](/docs/img/architecture_onehot.png)


[Implementacja](phisher/model/phisher_one_hot_model.py)

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