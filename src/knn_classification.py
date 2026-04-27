import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import StandardScaler

# slowniki
species_names_dict = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}
names_in_polish_dict = {
    'sepal length': 'Długość działki kielicha (cm)',
    'sepal width': 'Szerokość działki kielicha (cm)',
    'petal length': 'Długość płatka (cm)',
    'petal width': 'Szerokość płatka (cm)'
}

# zczytanie danych
df_train = pd.read_csv('data/data3_train.csv', header=None)
df_test = pd.read_csv('data/data3_test.csv', header=None)

# nazwy kolumn
df_train.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
df_test.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

# normalizacja danych
scaler = StandardScaler()

X_train = df_train[['sepal length', 'sepal width', 'petal length', 'petal width']]
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
Y_train = df_train['species']

X_test = df_test[['sepal length', 'sepal width', 'petal length', 'petal width']]
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
Y_test = df_test['species']


# funkcja zeby byly wartosci nad slupkami
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)


# algorytm knn

# wszystkie cechy
accuracy_scores = []
best_accuracy_score = 0
best_conf_matrix = []
best_k = 0

print("-" * 30)

for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    if accuracy > best_accuracy_score:
        best_accuracy_score = accuracy
        best_conf_matrix = conf_matrix
        best_k = k

    accuracy_scores.append((k, accuracy * 100))

# rysowanie wykresu dla wszystkich cech
accuracy_scores_df = pd.DataFrame(accuracy_scores, columns=['num_of_neighbors', 'score'])
plt.figure(figsize=(10, 6))

# rysowanie slupkow + zapisanie do zmiennej bars
bars = plt.bar(data=accuracy_scores_df, x='num_of_neighbors', height='score', color='skyblue', edgecolor='black')

# wywolanie tej naszej funkcji
add_labels(bars)

plt.xticks(np.arange(1, 16, step=1))
plt.yticks(np.arange(0, 110, step=10))
plt.ylim(0, 115)
plt.xlabel('k', fontsize=14)
plt.ylabel('Dokładność klasyfikacji w %', fontsize=14)
plt.title('Wszystkie cechy')
plt.show()

print('Macierz pomyłek:')
print(best_conf_matrix)
print(f'Dokładność: {best_accuracy_score * 100:.1f}%')
print("k: " + str(best_k))

# pary cech, druga kropka z punktu 2
for x_label, y_label in combinations(df_train.columns[:-1], 2):
    accuracy_scores = []
    best_accuracy_score = 0
    best_conf_matrix = []
    best_k = 0

    pair_name = names_in_polish_dict[x_label] + ' + ' + names_in_polish_dict[y_label]
    print("-" * 30)
    print(pair_name)

    for k in range(1, 16):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train[[x_label, y_label]], Y_train)
        Y_pred = knn.predict(X_test[[x_label, y_label]])

        accuracy = accuracy_score(Y_test, Y_pred)
        conf_matrix = confusion_matrix(Y_test, Y_pred)

        if accuracy > best_accuracy_score:
            best_accuracy_score = accuracy
            best_conf_matrix = conf_matrix
            best_k = k

        accuracy_scores.append((k, accuracy * 100))

    accuracy_scores_df = pd.DataFrame(accuracy_scores, columns=['num_of_neighbors', 'score'])

    # rysowanie wykresu par
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data=accuracy_scores_df, x='num_of_neighbors', height='score', color='lightgreen', edgecolor='black')

    # napisy
    add_labels(bars)

    plt.yticks(np.arange(0, 110, step=10))
    plt.xticks(np.arange(1, 16, step=1))
    plt.ylim(0, 115)
    plt.xlabel('k', fontsize=14)
    plt.ylabel('Dokładność klasyfikacji w %', fontsize=14)
    plt.title(pair_name)
    plt.show()

    print('Macierz pomyłek:')
    print(best_conf_matrix)
    print(f'Dokładność: {best_accuracy_score * 100:.1f}%')
    print("k: " + str(best_k))