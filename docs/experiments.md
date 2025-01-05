# Metodologia

Wykorzystano dwa zbiory danych:  
- [Web Page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset?resource=download)  
- [PHIUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).  

Łącznie zbiory zawierają **247225 próbek**, z podziałem:  
- **Label 0 (phishing)**: 106660 (43.14%)  
- **Label 1 (bezpieczny)**: 140565 (56.86%).  

Przed przetwarzaniem dane zredukowano do dwóch kolumn: `url` i `status`. Kolumna `url` została sprowadzona do hostname, usuwając protokoły i ścieżki.

Stworzono dwa modele:  
- **OneHot** ([opis](./onehot.md)),  
- **Embeddings** ([opis](./embeddings.md)).  

Każdy z modeli używa dedykowanego preprocessingu, gdzie zostały one opisane w wyżej przedstawionych plikach md. 

Wyniki eksperymentów można znaleźć w [Weights & Biases](https://wandb.ai/bartosz-kosinski-b-warsaw-university-of-technology/phisher?nw=nwuserbartoszkosinskib).


### Porównanie wyników z literaturą

#### Wprowadzenie
W dwóch artykułach naukowych wykorzystano różne podejścia do detekcji phishingu:

1. **Paper 1**: [An Effective Phishing Detection Model Based on Character Level Convolutional Neural Network from URL](https://doi.org/10.3390/electronics9091514)
   - Zastosowano embeddingi znaków URL.
   - Wyniki:

   | **Model**                | **F1 (%)** | **Precyzja (%)** | **Czułość (%)** | **Dokładność (%)** |
   |--------------------------|------------|------------------|----------------|-------------------|
   | paper_model_embeddings   | 98.56      | 98.55            | 98.62          | 98.58            |

2. **Paper 2**: [A Deep Learning-Based Phishing Detection System Using CNN, LSTM, and LSTM-CNN](https://doi.org/10.3390/electronics12010232)
   - Wykorzystano pełne dane z datasetu.
   - Wyniki:

   | **Model**                | **F1 (%)** | **Precyzja (%)** | **Czułość (%)** | **Dokładność (%)** |
   |--------------------------|------------|------------------|----------------|-------------------|
   | paper_model_cnn          | 99.2      | 99            | 99.2          | 99.2            |

#### Porównanie wyników
Zestawienie wyników naszych modeli z wynikami z literatury:

| **Model**                | **F1 (%)** | **Precyzja (%)** | **Czułość (%)** | **Dokładność (%)** |
|--------------------------|------------|------------------|----------------|-------------------|
| paper_model_embeddings   | 98.56      | 98.55            | 98.62          | 98.58            |
| paper_model_cnn          | 99.2       | 99               | 99.2           | 99.2             |
| Nasz model (Embeddings)   | 99.39      | 99.23            | 99.57          | 99.35            |
| Nasz model (OneHot)     | 98.59      | 98.59            | 98.68          | 98.49            |

#### Analiza
Wyniki naszych modeli wskazują na wysoką skuteczność obu podejść (embeddings i one-hot), ale model z embeddingami przewyższa zarówno nasz model one-hot, jak i modele z literatury.

Paper 1: Model bazujący na embeddingach znaków osiągnął solidne wyniki (F1 = 98.56%), jednak nasze podejście embeddingowe poprawiło ten wynik, osiągając F1 = 99.39%.

Paper 2: Model CNN z pełnymi danymi uzyskał najwyższe wyniki (F1 = 99.2%), jednak nasze embeddingi były porównywalne, przewyższając pod względem precyzji (99.23%) i czułości (99.57%).

Podsumowując, nasz model embeddingowy wykazuje przewagę nad literaturą dzięki lepszemu wykorzystaniu charakterystyk danych URL.




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