import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import math

#wczytywanie danych
species_dict = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}
datafile = pd.read_csv("data/data1.csv", header=None, sep=',')
datafile.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']


#liczenie gatunkow z sortowaniem + wyswietlanie
species_counts = datafile['species'].value_counts().sort_index()
total = len(datafile)

print("\nTABELA 1 – Liczebność gatunków")
print("Gatunek\t\tLiczebność (%)")
for idx, count in species_counts.items():
    percent = round(count / total * 100, 1)
    print(f"{species_dict[idx]:<10}\t{count} ({percent}%)")
print(f"Razem\t\t{total} (100%)")

#miary rozkladu
columns = ['sepal length', 'sepal width', 'petal length', 'petal width']

print("\nTABELA 2 – Miary rozkładu")
print(f"{'Cecha':<18} {'Min':>5}    {'Śr. arytm. (± odch. stand.)':>25} {'Mediana (Q1 - Q3)':>22} {'Maks':>6}")

for col in columns:
    data = datafile[col]
    minimum = round(data.min(), 2)
    maximum = round(data.max(), 2)
    #srednia art
    mean = round(statistics.mean(data), 2)
    #odchylenie standardowe
    std = round(statistics.stdev(data), 2)
    #mediana
    median = round(statistics.median(data), 2)
    #kwartyle q1 i q3, funckja percentile sortuje dane rosnaco automatycznie
    q1 = round(np.percentile(data, 25), 2)
    q3 = round(np.percentile(data, 75), 2)

    print(
        f"{col:<18} {minimum:>6.2f} {mean:>6.2f} (±{std:<4.2f}) {median:>24.2f} ({q1:<4.2f}-{q3:<4.2f}) {maximum:>7.2f}")

species_names = [v for k, v in species_dict.items()]

plt.rcParams['font.size'] = 12

#dlugosc dzialki kielicha - histogram
#+0.5, bo np.arrange (ciag liczb) nie wlicza ostatniej wartosci i moze uciac
value_divs = np.arange(math.floor(datafile['sepal length'].min()),
                       math.ceil(datafile['sepal length'].max()) + 0.5, step=0.5)
datafile['sepal length'].hist(edgecolor='black', bins=value_divs, grid=False)
plt.xlabel('Długość (cm)', fontsize=14)
plt.ylabel('Liczebność', fontsize=14)
plt.title('Długość działki kielicha', fontsize=20)
plt.xticks(value_divs)
plt.ylim(0, 35)
plt.show()

#dlugosc dzialki kielicha - pudelkowy
datafile.groupby('species')[['sepal length']].boxplot(subplots=False, grid=False,
                                                medianprops=dict(color="red"))
plt.xticks([1, 2, 3], labels=species_names)
plt.xlabel('Gatunek', fontsize=14)
plt.ylabel('Długość (cm)', fontsize=14)
plt.yticks(value_divs)
plt.show()

#szerokosc dzialki kielicha - histogram
value_divs = np.arange(math.floor(datafile['sepal width'].min()),
                       math.ceil(datafile['sepal width'].max()), step=0.5)
datafile['sepal width'].hist(edgecolor='black', bins=value_divs, grid=False)
plt.xlabel('Szerokość (cm)', fontsize=14)
plt.ylabel('Liczebność', fontsize=14)
plt.title('Szerokość działki kielicha', fontsize=20)
plt.xticks(np.arange(2, 5, step=0.5))
plt.ylim(0, 70)
plt.show()

#szerokosc dzialki kielicha - pudelkowy
datafile.groupby('species')[['sepal width']].boxplot(subplots=False, grid=False,
                                               medianprops=dict(color="red"))
plt.xticks([1, 2, 3], labels=species_names)
plt.xlabel('Gatunek', fontsize=14)
plt.ylabel('Szerokość (cm)', fontsize=14)
plt.yticks(value_divs)
plt.show()

#dlugosc platka - histogram
value_divs = np.arange(math.floor(datafile['petal length'].min()),
                       math.ceil(datafile['petal length'].max()) + 0.5, step=0.5)
datafile['petal length'].hist(edgecolor='black', bins=value_divs, grid=False)
plt.xlabel('Długość (cm)', fontsize=14)
plt.ylabel('Liczebność', fontsize=14)
plt.title('Długość płatka', fontsize=20)
plt.xticks(np.arange(1, 8, step=1))
plt.ylim(0, 30)
plt.show()

#dlugosc platka - pudelkowy
datafile.groupby('species')[['petal length']].boxplot(subplots=False, grid=False,
                                                medianprops=dict(color="red"))
plt.xticks([1, 2, 3], labels=species_names)
plt.xlabel('Gatunek', fontsize=14)
plt.ylabel('Długość (cm)', fontsize=14)
plt.yticks(np.arange(1, 8, step=1))
plt.show()

#szerokosc platka - histogram
value_divs = np.arange(math.floor(datafile['petal width'].min()),
                       math.ceil(datafile['petal width'].max()) + 0.5, step=0.5)
datafile['petal width'].hist(edgecolor='black', bins=value_divs, grid=False)
plt.xlabel('Szerokość (cm)', fontsize=14)
plt.ylabel('Liczebność', fontsize=14)
plt.title('Szerokość płatka', fontsize=20)
plt.xticks(value_divs)
plt.ylim(0, 50)
plt.show()

#szerokosc platka - pudelkowy
datafile.groupby('species')[['petal width']].boxplot(subplots=False, grid=False,
                                               medianprops=dict(color="red"))
plt.xticks([1, 2, 3], labels=species_names)
plt.xlabel('Gatunek', fontsize=14)
plt.ylabel('Szerokość (cm)', fontsize=14)
plt.yticks(value_divs)
plt.show()

#zad 4

#lista par cech (x,y)
feature_pairs = [
    ('sepal length', 'sepal width'),
    ('sepal length', 'petal length'),
    ('sepal length', 'petal width'),
    ('sepal width', 'petal length'),
    ('sepal width', 'petal width'),
    ('petal length', 'petal width')
]

#polskie nazwy do podpisow osi
features_polish = {
    'sepal length': 'Długość działki kielicha (cm)',
    'sepal width': 'Szerokość działki kielicha (cm)',
    'petal length': 'Długość płatka (cm)',
    'petal width': 'Szerokość płatka (cm)'
}

plt.rcParams['font.size'] = 10

#przejscie po kazdej parze cech
for x_feat, y_feat in feature_pairs:
    x = datafile[x_feat]
    y = datafile[y_feat]

    #wspolczynnik korelacji Pearsona
    r = np.corrcoef(x, y)[0, 1]
    r_rounded = round(r, 2)

    #wspołczynniki regresji liniowej (a – nachylenie, b – wyraz wolny)
    a, b = np.polyfit(x, y, 1)
    a_rounded = round(a, 1)
    b_rounded = round(b, 1)

    #generowanie prostej regresji
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a * x_line + b

    #rysowanie wykresu
    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, color='steelblue')
    plt.plot(x_line, y_line, color='red')
    plt.xlabel(features_polish[x_feat])
    plt.ylabel(features_polish[y_feat])
    plt.title(f"r = {r_rounded};  y = {a_rounded}x + {b_rounded}")
    plt.tight_layout()
    plt.show()