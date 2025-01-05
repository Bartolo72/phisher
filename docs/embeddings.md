# Architektura modelu
![Architektura](/docs/img/architecture_embeddings.png)


[Implementacja](phisher/model/phisher_embeddings_model.py)

# Preprocessing danych

Przetwarzanie danych w `PhishingEmbeddingDataset` obejmuje:

1. **Odczyt danych**: Plik CSV z URL-ami i etykietami (`1` - bezpieczny, `0` - phishingowy).
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