# Metodologia

Wykorzystano dwa zbiory danych:  
- [Web Page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset?resource=download)  
- [PHIUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).  

Łącznie zbiory zawierają **247225 próbek**, z podziałem:  
- **Label 0 (phishing)**: 106660 (43.14%)  
- **Label 1 (bezpieczny)**: 140565 (56.86%).  

Przed przetwarzaniem dane zredukowano do dwóch kolumn: `url` i `status`. Kolumna `url` została sprowadzona do hostname, usuwając protokoły i ścieżki.

Stworzono dwa modele:  
- **OneHot** ([opis](docs/onehot.md)),  
- **Embeddings** ([opis](docs/embeddings.md)).  

Każdy z modeli używa dedykowanego preprocessingu, gdzie zostały one opisane w wyżej przedstawionych notatnikach. 

Wyniki eksperymentów można znaleźć w [Weights & Biases](https://wandb.ai/bartosz-kosinski-b-warsaw-university-of-technology/phisher?nw=nwuserbartoszkosinskib).


### Porównanie wyników z literaturą

#### Wprowadzenie
W dwóch artykułach naukowych wykorzystano różne podejścia do detekcji phishingu:

1. **Paper 1**: [An Effective Phishing Detection Model Based on Character Level Convolutional Neural Network from URL](https://doi.org/10.3390/electronics9091514)
   - Zastosowano embeddingi znaków URL.
   - Wyniki:

   | **Model**                | **F1 (%)** | **Precyzja (%)** | **Czułość (%)** | **Dokładność (%)** |
   |--------------------------|------------|------------------|----------------|-------------------|
   | paper_model_embeddings   | 97.77      | 97.91            | 97.62          | 97.75            |

2. **Paper 2**: [A Deep Learning-Based Phishing Detection System Using CNN, LSTM, and LSTM-CNN](https://doi.org/10.3390/electronics12010232)
   - Wykorzystano pełne dane z datasetu.
   - Wyniki:

   | **Model**                | **F1 (%)** | **Precyzja (%)** | **Czułość (%)** | **Dokładność (%)** |
   |--------------------------|------------|------------------|----------------|-------------------|
   | paper_model_cnn          | 98.56      | 98.55            | 98.62          | 98.58            |

#### Porównanie wyników
Zestawienie wyników naszych modeli z wynikami z literatury:

| **Model**                | **F1 (%)** | **Precyzja (%)** | **Czułość (%)** | **Dokładność (%)** |
|--------------------------|------------|------------------|----------------|-------------------|
| paper_model_embeddings   | 97.77      | 97.91            | 97.62          | 97.75            |
| paper_model_cnn          | 98.56      | 98.55            | 98.62          | 98.58            |
| Nasz model (Embeddings)   | 99.39      | 99.23            | 99.57          | 99.35            |
| Nasz model (OneHot)     | 98.59      | 98.59            | 98.68          | 98.49            |

#### Analiza
Nasze modele osiągnęły wyższe wyniki niż modele opisane w literaturze, co więcej udało nam się poprawić wyniki dla modelu opartego o nie tylko zanurzenia znaków, co sugeruje skuteczność zastosowanego preprocessingu i architektur sieci neuronowych.




# LLM

Za pomocą obecnej wersji modelu językowego ChatGPT wygenerowano **10,000 próbek** najbardziej popularnych metod typosquattingu dla znanych domen internetowych.

Przetestowano wcześniej wytrenowany model na datasetach opisanych w sekcji [Metodologia](#metodologia). 

Uzyskano następujące wyniki:

Zestawienie wyników dla modeli one-hot i embedding, przetestowanych na danych generowanych za pomocą ChatGPT:

| **Model**       | **Dokładność (%)** | **F1 (%)** | **Precyzja (%)** | **Czułość (%)** |
|------------------|------------------------------|------------|-----------------------------|-------------------------|
| One-hot          | 98.01                       | 98.12      | 98.42                       | 97.93                  |
| Embeddings       | 99.17                       | 99.22      | 99.35                       | 99.13                  |

Wyniki jednoznacznie wskazują na wyższą skuteczność modelu wykorzystującego embeddingi w porównaniu z modelem one-hot.

Aby zewaluować wyniki na opisanym zbiorze należy uzupełnić komendę o argument: `data=llm`