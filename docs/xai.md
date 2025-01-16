# Wyjaśnialność modeli - XAI

Stworzone modele osiągnęły bardzo dobre wyniki, co szczegółowo opisano w sekcji [eksperymenty](./experiments.md). W ramach dalszych ulepszeń projektu zdecydowaliśmy się na implementację narzędzi umożliwiających lepsze zrozumienie procesów decyzyjnych naszego modelu.

Obecnie skupiliśmy się na modelu opartym o embeddingi, ponieważ uzyskał on lepsze rezultaty i działa bardziej efektywnie.

## Wyjaśnialność modelu z embeddingami

W ramach pracy nad wyjaśnialnością zaimplementowaliśmy dwa narzędzia:
- [Grad-CAM](https://jacobgil.github.io/pytorch-gradcam-book/introduction.html) – metoda inspirowana dziedziną wizji komputerowej, adaptowana do naszych potrzeb,
- [Lime](https://github.com/marcotcr/lime) – ogólne narzędzie służące do wyjaśnialności klasyfikatorów dowolnego typu.

## Przeprowadzone eksperymenty – metodyka

Wykorzystaliśmy ogólnodostępną [listę niebezpiecznych domen](https://hole.cert.pl/domains/v2/domains.txt) oferowaną przez cert.pl. Wybraliśmy 5 niebezpiecznych adresów URL, na których przetestowaliśmy nasze narzędzia:


```
apple-search.world
apple.com-tracked.services
apple.find-lost.click
apple.findmy-auth.click
apple.tracker-liveview.support
```

Celem eksperymentów było podkreślenie tych fragmentów adresów URL, które w największym stopniu wpłynęły na decyzję modelu.

### Grad-CAM

Przestrzeń embeddingów została przedstawiona jako macierz o określonych wymiarach. Inspirując się zaawansowanymi metodami wyjaśnialności stosowanymi w wizji komputerowej, przetestowaliśmy rozwiązanie [Grad-CAM](https://jacobgil.github.io/pytorch-gradcam-book/introduction.html). Celem było sprawdzenie, czy reprezentacja przestrzeni embeddingów może zastąpić obraz, zachowując swoją wyjaśnialność. 

> [!NOTE]  
> Grad-CAM wymaga wybrania warstwy konwolucyjnej w celu wizualizacji filtrów wyuczonych przez model. W przeprowadzonych eksperymentach wybraliśmy pierwszą warstwę konwolucyjną. Szczegółowy opis architektury modelu znajduje się [tutaj](./embeddings.md#architektura-modelu).



W [notatniku](../notebooks/xai_grad_cam.ipynb) znajduje się przykład użycia metody Grad-CAM dla przykładowego złośliwego adresu URL.

Implementacja została umieszczona [tutaj](../phisher/xai/gradcam.py).

### Lime

Lime, jako narzędzie wyjaśnialności klasyfikatorów dowolnego typu, pozwala lepiej zrozumieć działanie badanego modelu. W dużym uproszczeniu, Lime działa poprzez maskowanie kolejnych elementów danych wejściowych, przeprowadzanie na nich predykcji oraz porównywanie tych predykcji z oryginalnym wynikiem w celu określenia najbardziej znaczących elementów.

W [notatniku](../notebooks/xai_lime.ipynb) znajduje się przykład użycia rozwiązania Lime dla przykładowego złośliwego adresu URL.

Implementacja znajduje się [tutaj](../phisher/xai/phish_lime.py).

### Wyniki

| **Adres URL**               | **Predykcja modelu**       | **Grad-CAM**                     | **Lime**                     |
|-----------------------------|----------------------------|-----------------------------------|------------------------------|
| `apple-search.world`        | Phishing (pewność 55%)    | ![1](/docs/img/xai/gradcam/1.png)| ![1](/docs/img/xai/lime/1.png)|
| `apple.com-tracked.services`| Phishing (pewność 100%)   | ![2](/docs/img/xai/gradcam/2.png)| ![2](/docs/img/xai/lime/2.png)|
| `apple.find-lost.click`     | Phishing (pewność 100%)   | ![3](/docs/img/xai/gradcam/3.png)| ![3](/docs/img/xai/lime/3.png)|
| `apple.findmy-auth.click`   | Phishing (pewność 100%)   | ![4](/docs/img/xai/gradcam/4.png)| ![4](/docs/img/xai/lime/4.png)|
| `apple.tracker-liveview.support` | Phishing (pewność 100%) | ![5](/docs/img/xai/gradcam/5.png)| ![5](/docs/img/xai/lime/5.png)|

#### Kolory klas w Lime:
![](https://img.shields.io/badge/Phishing%20URL-1f77b4)
![](https://img.shields.io/badge/Safe%20URL-fb7d0f)

#### Kolory w Grad-CAM:
![](https://img.shields.io/badge/Phishing%20URL-0000ff)

### Wnioski

Z naszych obserwacji wynika, że architektura modelu ma kluczowe znaczenie dla skuteczności implementacji Grad-CAM. Jak pokazano w sekcji wyników, metoda ta pozwala zrozumieć znaczenie wyuczonych filtrów w warstwach konwolucyjnych. Niestety, nasz model zawiera także warstwy liniowe, co ogranicza przydatność tego narzędzia w pełnym rozumieniu wyjaśnialności problemu klasyfikacji niebezpiecznych adresów URL. Może to prowadzić do błędnych interpretacji.

W przypadku Lime zauważono pewne ograniczenia wizualne – liczba znaków wskazujących na daną klasę czasami nie odpowiada rzeczywistej klasie przewidzianej przez model. Przykładami są adresy `apple.find-lost.click` oraz `apple.tracker-liveview.support`. Mimo to Lime okazuje się bardziej użyteczne w analizie wyjaśnialności naszego modelu. W wielu przypadkach wyraźnie wskazuje fragmenty adresu URL, które model uznał za kluczowe dla oceny zagrożenia, co może być pomocne w identyfikacji potencjalnych prób typosquattingu.
