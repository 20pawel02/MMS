import numpy as np

# 1. Wczytanie danych z pliku
dane_cale = []
with open("lab1_dane.txt", 'r') as fi:
    for line in fi:
        if line.split():
            line = [float(x) for x in line.split()]
            dane_cale.append(line)

# 2. Rozdzielenie wierszy na 4 osobne zestawy danych (kolumny)
dane1, dane2, dane3, dane4 = [], [], [], []
for item in dane_cale:
    dane1.append(item[0])
    dane2.append(item[1])
    dane3.append(item[2])
    dane4.append(item[3])

# Zgrupowanie zestawów w jedną listę
zestawy = [dane1, dane2, dane3, dane4]

# 3. Obliczenie i wypisanie wszystkich statystyk dla każdego zestawu
for i, dane in enumerate(zestawy, start=1):
    print(f"--- ZESTAW {i} ---")
    
    # Obliczenia
    x_sr = round(np.mean(dane), 2)                # Średnia
    s = round(np.std(dane), 2)                    # Odch. std. populacji
    s_p = round(np.std(dane, ddof=1), 2)          # Odch. std. z próby
    minimum = round(np.min(dane), 2)              # Minimum
    Q1 = round(np.percentile(dane, 25), 2)        # Kwartyl 1 (25%)
    M = round(np.median(dane), 2)                 # Mediana (50%)
    Q3 = round(np.percentile(dane, 75), 2)        # Kwartyl 3 (75%)
    maksimum = round(np.max(dane), 2)             # Maksimum
    
    # Wypisanie wyników
    print(f"Średnia:             {x_sr}")
    print(f"Odch. std. Popul.:   {s}")
    print(f"Odch. std. z próby:  {s_p}")
    print(f"Min:                 {minimum}")
    print(f"Kwartyl 1 (Q1):      {Q1}")
    print(f"Mediana (M):         {M}")
    print(f"Kwartyl 3 (Q3):      {Q3}")
    print(f"Max:                 {maksimum}")
    print("-" * 30)