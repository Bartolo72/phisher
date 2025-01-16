
# phisher

Moduł stworzony na potrzeby projektu na przedmiocie GSN na Politechnice Warszawskiej.


# Instalacja
```
pip install -r requirements.txt
pip install -e .
```

# Użycie
Przykład pokazano w [notatniku](notebooks/phisher.ipynb).


# Dokumentacja
1. [Model OneHot](docs/onehot.md)
    - [Architektura modelu](docs/onehot.md#architektura-modelu)
    - [Preprocessing danych wejściowych](docs/onehot.md#preprocessing-danych)
    - [Trenowanie i ewaluacja](docs/onehot.md#trenowanie-i-ewaluacja)
    - [Ewaluacja checkpointu](docs/onehot.md#ewaluacja-checkpointu)
2. [Model Embedding](docs/embeddings.md)
    - [Architektura modelu](docs/embeddings.md#architektura-modelu)
    - [Preprocessing danych wejściowych](docs/embeddings.md#preprocessing-danych)
    - [Trenowanie i ewaluacja](docs/embeddings.md#trenowanie-i-ewaluacja)
    - [Ewaluacja checkpointu](docs/embeddings.md#ewaluacja-checkpointu)
3. [Eksperymenty](docs/experiments.md)
    - [Metodologia](docs/experiments.md#metodologia)
    - [Porównanie wyników z literaturą](docs/experiments.md#porównanie-wyników-z-literaturą)
    - [LLM](docs/experiments.md#LLM)
4. [XAI](docs/xai.md)
    - [Wyjaśnialność modelu z embeddingami](docs/xai.md#wyjaśnialność-modeli---xai)
    - [Przeprowadzone eksperymenty – metodyka](docs/xai.md#przeprowadzone-eksperymenty-–-metodyka)
    - [Grad-CAM](docs/xai.md#grad-cam)
    - [Lime](docs/xai.md#lime)
    - [Wyniki](docs/xai.md#wyniki)
    - [Wnioski](docs/xai.md#wnioski)