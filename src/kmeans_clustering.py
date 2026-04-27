import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# SŁOWNIKI
names_in_polish_dict = {
    'sepal length': 'Długość działki kielicha',
    'sepal width': 'Szerokość działki kielicha',
    'petal length': 'Długość płatka',
    'petal width': 'Szerokość płatka'
}

# kolory kolek na wykresie i ksztalt centroidow
custom_colors = ['red', 'green', 'blue']
centroid_marker = 'D'

# 1. WCZYTANIE DANYCH bez nagłówka
df = pd.read_csv('data/data2.csv', header=None)
# nazwy kolumn
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width']

# 2. NORMALIZACJA DANYCH
# Ten wzór na xscaled z dołu pdf, żeby mieć zakresy 0-1
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)
# Zamieniamy z powrotem na DataFrame, żeby było wygodnie używać nazw kolumn
df_scaled = pd.DataFrame(data_scaled, columns=df.columns)

# 3. ALGORYTM K-ŚREDNICH
# n_clusters=3 z polecenia, random_state to losowa liczba, zamrażamy ją, żeby wynik zawsze byłtaki sam
kmeans = KMeans(n_clusters=3, random_state=50)

# pętle itp, dostajemy w labels wynik przynależności skwiatków do grupy
labels = kmeans.fit_predict(df_scaled)

# współrzędne centroidów grup, z algorytmu mamy centroidy w skali 0-1, dlatego w kolejnej linijce wracamy z nimi na cm tak jak w poleceniu jest
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# 4. WYKRESY
# czcionka
plt.rcParams['font.size'] = 12

# pętla dla każdej kombinacji (łącznie jest 6)
print("Generowanie wykresów podziału na grupy")
for x_col, y_col in combinations(df.columns, 2):
    plt.figure(figsize=(8, 6))

    # centroidy
    # x_col i y_col to nazwy kolumn, musimy znaleźć ich indeksy, żeby pobrać dobre współrzędne centroidów
    x_idx = df.columns.get_loc(x_col)
    y_idx = df.columns.get_loc(y_col)

    # rysowanie kazdej grupy osobno, zeby mozna dac obwodki jak w pdf kmeans
    for group_id in range(3):
        # 1. kwiaty z konkretnej grupy
        mask = (labels == group_id)
        # bierzemy oryginalne dane bez normalizacji
        current_x = df.loc[mask, x_col]
        current_y = df.loc[mask, y_col]

        # kolor dla tej grupy
        color = custom_colors[group_id]

        # 2. rysujemy punkty (kwiatki)
        # facecolors='none' -> środek przezroczysty
        # edgecolors=color -> obwódka w kolorze grupy
        plt.scatter(current_x, current_y,
                    facecolors='none',
                    edgecolors=color,
                    label=f'Grupa {group_id}')

        # 3. Rysujemy centroid dla tej grupy
        # wypełniony, żeby było widać gdzie jest środek
        cx = centroids_original[group_id, x_idx]
        cy = centroids_original[group_id, y_idx]
        plt.scatter(cx, cy,
                    c=color,
                    marker=centroid_marker,
                    s=150,
                    linewidth=1)

    plt.title(f'Grupowanie K-Means: {names_in_polish_dict[x_col]} vs {names_in_polish_dict[y_col]}')
    plt.xlabel(f'{names_in_polish_dict[x_col]} (cm)')
    plt.ylabel(f'{names_in_polish_dict[y_col]} (cm)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# 5. METODA ŁOKCIA (Analiza WCSS)
# sprawdzamy, czy 3 grupy to na pewno dobry wybór
# sprawdzamy kiedy 'jakosc' czy 'zysk' z dodawania nowych grup spada, czyli kiedy na wykresie
# linia zaczyna sie wyplaszczac i juz tak gwaltownie nie spada, tam jest najlepsza ilosc grup
# teoretycznie im mniejsze wscc tym ardziej zwarte i podobne do siebie punkty, ale tak to zawsze wybieralibysmy
# najwiecej grup, dlatego szukamy tego idealnego momentu metoda lokcia
print("Generowanie wykresu metody łokcia")

wcss = []
iterations = []
k_range = range(2, 11)

for k in k_range:
    # w bibliotece sklearn WCSS nazywa się 'inertia_'
    temp_kmeans = KMeans(n_clusters=k, random_state=50)
    temp_kmeans.fit(df_scaled)

    # pobieramy WCSS (inertia_)
    val_wcss = temp_kmeans.inertia_
    wcss.append(val_wcss)

    # pobieramy liczbę iteracji (n_iter_)
    val_iter = temp_kmeans.n_iter_
    iterations.append(val_iter)

# Wykres 1: metoda Łokcia (WCSS)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1) # dzielimy okno na 2 części, to jest część 1
plt.plot(k_range, wcss, marker='o', linestyle='--', color='blue')
plt.xticks(k_range)
plt.title('Metoda Łokcia (WCSS)')
plt.xlabel('Liczba klastrów (k)')
plt.ylabel('WCSS')
plt.grid(True)

# Wykres 2: liczba iteracji
plt.subplot(1, 2, 2) # część 2
plt.bar(k_range, iterations, color='orange', edgecolor='black', alpha=0.7)
plt.xticks(k_range)
plt.title('Liczba iteracji do zbieżności')
plt.xlabel('Liczba klastrów (k)')
plt.ylabel('Iteracje')
plt.grid(True, axis='y')

# żeby wykresy na siebie nie nachodziły
plt.tight_layout()
plt.show()

print("\nTabela WCSS")
tabela_wynikow = pd.DataFrame({
    'Liczba klastrów (k)': k_range,
    'WCSS (Inertia)': wcss,
    'Liczba iteracji': iterations
})

print(tabela_wynikow.to_string(index=False))