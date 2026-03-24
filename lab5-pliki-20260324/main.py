import numpy as np
import scipy.stats
from scipy.stats import norm, poisson
import pandas as pd

# =====================================================================
# 1. WCZYTANIE DANYCH
# Zgodnie z poleceniem: dane.xlsx, cecha Masa dzieci w wieku 11.50-12.50
# =====================================================================

# ODKOMENTUJ PONIŻSZE LINIE, aby użyć pliku Excela (wymaga pip install pandas openpyxl)
# df = pd.read_excel('dane.xlsx')
# nazwa_kolumny = 'Masa dzieci w wieku 11.50-12.50' # podaj dokładną nazwę kolumny
# dane = df[nazwa_kolumny].dropna().tolist()

# TYMCZASOWE DANE (dla celów testowych, zastępujące wczytywanie z .txt)
# Jeśli używasz pliku txt z zajęć, zachowujemy Twoją logikę:
dane_sz = []
with open("dane_wys_lab2.txt", 'r') as fi:
    for line in fi:
        if line.split():
            line_vals = [float(x) for x in line.split()]
            dane_sz.append(line_vals)

dane = [number for item in dane_sz for number in item]

# =====================================================================
# PARAMETRY PODSTAWOWE I PARAMETRY DO TESTÓW
# =====================================================================
sr_p = np.mean(dane) # średnia z próby
std_p = np.std(dane) # odchylenie standardowe z próby

min_val = np.min(dane)
max_val = np.max(dane)
n = len(dane)
rozstep = max_val - min_val
k_rob = round(n**0.5, 1)

# Zakładamy szerokość przedziału (h) i początek pierwszej klasy (x01)
h = round(rozstep / n**0.5, 2) 
x01 = min_val - h/2 # Przesunięcie o pół kroku w lewo

print(f"Podsumowanie danych:\n min: {min_val}\n max: {max_val}\n n: {n}\n wyliczone h: {h}\n wyliczone x01: {x01}\n")

# Wybrane arbitralnie parametry do testowania (zgodnie z poleceniem)
srednia_test = 145.6
odch_stand_test = 6.2

# Wymagane poziomy istotności
poziomy_alfa = [0.01, 0.05, 0.1]

# =====================================================================
# FUNKCJE POMOCNICZE
# =====================================================================
def hipoteza_wnioski(test, kryt, alfa):
    if test < kryt:
        print(f"Dla alfa={alfa}: Brak podstaw do odrzucenia H0 (Statystyka: {test:.3f} < Krytyczna: {kryt:.3f})")
    else:
        print(f"Dla alfa={alfa}: Odrzucamy H0 na rzecz H1 (Statystyka: {test:.3f} >= Krytyczna: {kryt:.3f})")

def tworz_szereg_8(dane, h, x01):
    max_val = np.max(dane)
    przedzialy = []
    liczebnosci = []
    srodki = []

    x_left = x01
    x_right = x01 + h

    while x_left <= max_val: # zmieniono na <=
        n_elem = len([x for x in dane if (x > x_left and x <= x_right)])
        while n_elem < 8 and x_right < max_val + h: # zabezpieczenie pętli
            x_right += h
            n_elem = len([x for x in dane if (x > x_left and x <= x_right)])

        przedzialy.append([x_left, x_right])
        liczebnosci.append(n_elem)
        srodki.append((x_left + x_right) / 2)

        x_left = x_right
        x_right = x_left + h

    if liczebnosci[-1] < 8 and len(liczebnosci) > 1:
        przedzialy[-2][1] = przedzialy[-1][1]
        liczebnosci[-2] += liczebnosci[-1]
        srodki[-2] = sum(przedzialy[-2]) / 2
        przedzialy.pop()
        liczebnosci.pop()
        srodki.pop()

    return przedzialy, liczebnosci, srodki

# Przygotowanie szeregu
przedzialy, liczebnosci, srodki = tworz_szereg_8(dane, h, x01)

# =====================================================================
# A. TEST ZGODNOŚCI CHI-KWADRAT (ROZKŁAD NORMALNY)
# =====================================================================
print("\n--- A. TEST CHI-KWADRAT: ROZKŁAD NORMALNY ---")
print(f"H0: Rozkład cechy jest rozkładem normalnym N({srednia_test}, {odch_stand_test})")
print("H1: Rozkład cechy nie jest rozkładem normalnym")

# Prawdopodobieństwo teoretyczne (z domknięciem skrajnych ogonów do nieskończoności)
pi = []
for i, prz in enumerate(przedzialy):
    lewy_skraj = -np.inf if i == 0 else prz[0]
    prawy_skraj = np.inf if i == len(przedzialy) - 1 else prz[1]
    
    p_prawe = norm.cdf(prawy_skraj, loc=srednia_test, scale=odch_stand_test)
    p_lewe = norm.cdf(lewy_skraj, loc=srednia_test, scale=odch_stand_test)
    pi.append(p_prawe - p_lewe)

npi_norm = [n * p for p in pi]
chi2_norm = [(ni - npi)**2 / npi for npi, ni in zip(npi_norm, liczebnosci)]
testowa_norm = np.sum(chi2_norm)

r = 2 # liczba szacowanych parametrów (jeśli wpisane z palca to teoretycznie 0, ale zostawiamy konwencję)
st_swobody_norm = len(przedzialy) - r - 1

for alfa in poziomy_alfa:
    krytyczna_norm = scipy.stats.chi2.isf(alfa, st_swobody_norm)
    hipoteza_wnioski(testowa_norm, krytyczna_norm, alfa)


# =====================================================================
# B. TEST ZGODNOŚCI CHI-KWADRAT (ROZKŁAD POISSONA)
# =====================================================================
print("\n--- B. TEST CHI-KWADRAT: ROZKŁAD POISSONA ---")
print(f"H0: Rozkład cechy jest rozkładem Poissona z parametrem lambda={srednia_test}")
print("H1: Rozkład cechy nie jest rozkładem Poissona")

pi_poisson = []
for i, prz in enumerate(przedzialy):
    # Poisson jest dyskretny, przybliżamy różnicą dystrybuant
    p_prawe = poisson.cdf(prz[1], mu=srednia_test)
    p_lewe = poisson.cdf(prz[0], mu=srednia_test) if i > 0 else 0 # od 0 dla pierwszej klasy
    if i == len(przedzialy) - 1:
        p_prawe = 1.0 # Ostatnia klasa do nieskończoności
    pi_poisson.append(p_prawe - p_lewe)

npi_poisson = [n * p for p in pi_poisson]
chi2_poisson = [(ni - npi)**2 / npi for npi, ni in zip(npi_poisson, liczebnosci)]
testowa_poisson = np.sum(chi2_poisson)

st_swobody_poisson = len(przedzialy) - 1 - 1 # r=1 dla średniej

for alfa in poziomy_alfa:
    krytyczna_poisson = scipy.stats.chi2.isf(alfa, st_swobody_poisson)
    hipoteza_wnioski(testowa_poisson, krytyczna_poisson, alfa)


# =====================================================================
# C. TEST ZGODNOŚCI LAMBDA KOŁMOGOROWA (ROZKŁAD NORMALNY)
# =====================================================================
print("\n--- C. TEST LAMBDA KOŁMOGOROWA: ROZKŁAD NORMALNY ---")
print(f"H0: Rozkład cechy jest rozkładem normalnym N({srednia_test}, {odch_stand_test})")
print("H1: Rozkład cechy nie jest rozkładem normalnym")

# Uproszczone empiryczne wyliczenie dystrybuanty na danych posortowanych
# K-S działa najlepiej na danych niepogrupowanych!
dane_sorted = np.sort(dane)
dystr_empiryczna = np.arange(1, n+1) / n
F0_ks = norm.cdf(dane_sorted, loc=srednia_test, scale=odch_stand_test)

# Szukamy największej różnicy
roznice_ks = np.abs(F0_ks - dystr_empiryczna)
D_stat = np.max(roznice_ks)
testowa_K = np.sqrt(n) * D_stat

# Wartości krytyczne z rozkładu Kołmogorowa (aproksymacja)
# 0.01 -> 1.63, 0.05 -> 1.36, 0.1 -> 1.22
krytyczne_K_dict = {0.01: 1.63, 0.05: 1.36, 0.1: 1.22}

for alfa in poziomy_alfa:
    krytyczna_K = krytyczne_K_dict[alfa]
    hipoteza_wnioski(testowa_K, krytyczna_K, alfa)


# =====================================================================
# REGUŁA 3 SIGMA
# =====================================================================
print("\n--- REGUŁA 3 SIGMA ---")
dolna_granica = srednia_test - 3 * odch_stand_test
gorna_granica = srednia_test + 3 * odch_stand_test

w_przedziale = len([x for x in dane if dolna_granica <= x <= gorna_granica])
procent_w_przedziale = (w_przedziale / n) * 100

print(f"Przedział 3 sigma: [{dolna_granica:.2f}, {gorna_granica:.2f}]")
print(f"Liczba obserwacji w przedziale: {w_przedziale} z {n} ({procent_w_przedziale:.2f}%)")

if procent_w_przedziale >= 99.0:
    print("Wniosek: Dane spełniają regułę 3 sigma (blisko 99.7% danych w przedziale), co wskazuje na zgodność z rozkładem normalnym.")
else:
    print("Wniosek: Dane NIE spełniają idealnie reguły 3 sigma. Oznacza to potencjalne różnice względem rozkładu normalnego.")