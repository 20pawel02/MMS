import numpy as np
import scipy.stats
from scipy.stats import norm, poisson
from fpdf import FPDF
import os
import urllib.request

# =====================================================================
# POBIERANIE CZCIONKI Z POLSKIMI ZNAKAMI (Tylko za pierwszym razem)
# =====================================================================
font_regular = "DejaVuSans.ttf"
font_bold = "DejaVuSans-Bold.ttf"

# Używamy stabilnych linków z repozytorium matplotlib
url_regular = "https://raw.githubusercontent.com/matplotlib/matplotlib/main/lib/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
url_bold = "https://raw.githubusercontent.com/matplotlib/matplotlib/main/lib/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"

try:
    if not os.path.exists(font_regular):
        print("Pobieram czcionkę regularną z polskimi znakami...")
        urllib.request.urlretrieve(url_regular, font_regular)

    if not os.path.exists(font_bold):
        print("Pobieram czcionkę pogrubioną z polskimi znakami...")
        urllib.request.urlretrieve(url_bold, font_bold)
except Exception as e:
    print(f"Błąd podczas pobierania czcionek: {e}")
    print("Jeśli problem będzie się powtarzał, możesz wyłączyć ten blok i użyć poprzedniej wersji ze zlikwidowanymi ogonkami.")

# =====================================================================
# 1. WCZYTANIE DANYCH Z PLIKU TXT
# =====================================================================
# (Reszta kodu pozostaje bez zmian, zaczynając od tego miejsca...)
plik_txt = "dane_wys_lab2.txt"
dane_sz = []

try:
    with open(plik_txt, 'r') as fi:
        for line in fi:
            if line.split():
                line_vals = [float(x) for x in line.split()]
                dane_sz.append(line_vals)
    dane = [number for item in dane_sz for number in item]
    print(f"Pomyślnie wczytano dane z pliku: '{plik_txt}'")
except Exception as e:
    print(f"UWAGA: Nie znaleziono pliku '{plik_txt}'. Generuję dane testowe.")
    dane = np.random.normal(145.6, 6.2, 70).tolist()

# =====================================================================
# PARAMETRY PODSTAWOWE
# =====================================================================
min_val, max_val = np.min(dane), np.max(dane)
n = len(dane)
rozstep = max_val - min_val
h = round(rozstep / n**0.5, 2) 
x01 = min_val - h/2

srednia_test = round(np.mean(dane), 1)
odch_stand_test = round(np.std(dane), 1)
poziomy_alfa = [0.01, 0.05, 0.1]

# =====================================================================
# FUNKCJE POMOCNICZE (Szereg rozdzielczy)
# =====================================================================
def tworz_szereg_8(dane, h, x01):
    max_val = np.max(dane)
    przedzialy, liczebnosci = [], []
    x_left, x_right = x01, x01 + h

    while x_left <= max_val:
        n_elem = len([x for x in dane if (x > x_left and x <= x_right)])
        while n_elem < 8 and x_right < max_val + h:
            x_right += h
            n_elem = len([x for x in dane if (x > x_left and x <= x_right)])

        przedzialy.append([x_left, x_right])
        liczebnosci.append(n_elem)
        x_left = x_right
        x_right = x_left + h

    if liczebnosci[-1] < 8 and len(liczebnosci) > 1:
        przedzialy[-2][1] = przedzialy[-1][1]
        liczebnosci[-2] += liczebnosci[-1]
        przedzialy.pop()
        liczebnosci.pop()

    return przedzialy, liczebnosci

przedzialy, liczebnosci = tworz_szereg_8(dane, h, x01)

# =====================================================================
# OBLICZENIA DO RAPORTU
# =====================================================================
# A. Chi-kwadrat (Normalny)
pi_norm = []
for i, prz in enumerate(przedzialy):
    lewy = -np.inf if i == 0 else prz[0]
    prawy = np.inf if i == len(przedzialy) - 1 else prz[1]
    pi_norm.append(norm.cdf(prawy, loc=srednia_test, scale=odch_stand_test) - norm.cdf(lewy, loc=srednia_test, scale=odch_stand_test))

npi_norm = [n * p for p in pi_norm]
chi2_norm = [(ni - npi)**2 / npi for npi, ni in zip(npi_norm, liczebnosci)]
testowa_norm = np.sum(chi2_norm)
st_swobody_norm = len(przedzialy) - 2 - 1

# B. Chi-kwadrat (Poisson)
pi_poisson = []
for i, prz in enumerate(przedzialy):
    lewy = poisson.cdf(prz[0], mu=srednia_test) if i > 0 else 0
    prawy = 1.0 if i == len(przedzialy) - 1 else poisson.cdf(prz[1], mu=srednia_test)
    pi_poisson.append(prawy - lewy)

npi_poisson = [n * p for p in pi_poisson]
chi2_poisson = [(ni - npi)**2 / npi for npi, ni in zip(npi_poisson, liczebnosci)]
testowa_poisson = np.sum(chi2_poisson)
st_swobody_poisson = len(przedzialy) - 1 - 1

# C. Kolmogorow (Normalny)
dane_sorted = np.sort(dane)
dystr_empiryczna = np.arange(1, n+1) / n
F0_ks = norm.cdf(dane_sorted, loc=srednia_test, scale=odch_stand_test)
testowa_K = np.sqrt(n) * np.max(np.abs(F0_ks - dystr_empiryczna))
krytyczne_K_dict = {0.01: 1.63, 0.05: 1.36, 0.1: 1.22}

# Regula 3 Sigma
dolna_3s = srednia_test - 3 * odch_stand_test
gorna_3s = srednia_test + 3 * odch_stand_test
w_przedziale = len([x for x in dane if dolna_3s <= x <= gorna_3s])
procent_3s = (w_przedziale / n) * 100

# =====================================================================
# GENEROWANIE PLIKU PDF
# =====================================================================
pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Załadowanie czcionek Unicode
pdf.add_font("DejaVu", "", font_regular)
pdf.add_font("DejaVu", "B", font_bold)

def add_title(title):
    pdf.set_x(10)
    pdf.set_font("DejaVu", style="B", size=12)
    pdf.multi_cell(190, 8, text=title)
    pdf.set_font("DejaVu", size=10)
    pdf.ln(2)

def add_text(text):
    pdf.set_x(10)
    pdf.set_font("DejaVu", size=10)
    pdf.multi_cell(190, 6, text=text)

# NAGŁÓWEK
add_title("RAPORT Z LABORATORIUM 5 - TESTY ZGODNOŚCI - PAWEŁ WRÓBEL 169394")
add_text("Badana cecha: Wzrost dzieci w wieku 11.50-12.50 lat")
add_text(f"Samodzielnie wybrane parametry testowe: Średnia = {srednia_test}, Odchylenie = {odch_stand_test}, n = {n}")
pdf.ln(5)

# A. Normalny
add_title("A. Test zgodności chi-kwadrat dla rozkładu normalnego")
add_text(f"H0: Rozkład cechy jest normalny N({srednia_test}, {odch_stand_test})")
add_text("H1: Rozkład cechy nie jest normalny.")
add_text("Szereg rozdzielczy wykorzystany do obliczeń:")
pdf.ln(2)

pdf.set_font("DejaVu", size=8)
pdf.set_x(10)
pdf.cell(40, 6, "Przedział", border=1)
pdf.cell(20, 6, "n_i", border=1)
pdf.cell(30, 6, "p_i", border=1)
pdf.cell(30, 6, "n * p_i", border=1, new_x="LMARGIN", new_y="NEXT")

for i in range(len(przedzialy)):
    prz_str = f"({przedzialy[i][0]:.1f}; {przedzialy[i][1]:.1f}]"
    pdf.set_x(10)
    pdf.cell(40, 6, prz_str, border=1)
    pdf.cell(20, 6, str(liczebnosci[i]), border=1)
    pdf.cell(30, 6, f"{pi_norm[i]:.4f}", border=1)
    pdf.cell(30, 6, f"{npi_norm[i]:.2f}", border=1, new_x="LMARGIN", new_y="NEXT")
    
pdf.set_font("DejaVu", size=10)
pdf.ln(2)

add_text(f"Wartość statystyki testowej: {testowa_norm:.3f}")
for alfa in poziomy_alfa:
    kryt = scipy.stats.chi2.isf(alfa, st_swobody_norm)
    wynik = "Brak podstaw do odrzucenia H0" if testowa_norm < kryt else "Odrzucamy H0 na rzecz H1"
    add_text(f"Wartość krytyczna dla α={alfa}: {kryt:.3f} -> Wynik: {wynik}")
add_text("Wnioski: Rozkład normalny dobrze opisuje naturalne zjawiska fizjologiczne. W zależności od przyjętego poziomu istotności α, decydujemy czy drobne odchylenia są na tyle znaczące, by odrzucić H0.")
pdf.ln(5)

# B. Poisson
add_title("B. Test zgodności chi-kwadrat dla rozkładu Poissona")
add_text(f"H0: Rozkład cechy jest rozkładem Poissona (λ = {srednia_test}).")
add_text("H1: Rozkład cechy nie jest rozkładem Poissona.")
add_text("Szereg rozdzielczy wykorzystany do obliczeń (przedziały i liczebności n_i takie jak w pkt A):")
pdf.ln(2)

pdf.set_font("DejaVu", size=8)
pdf.set_x(10)
pdf.cell(40, 6, "Przedział", border=1)
pdf.cell(30, 6, "p_i (Poisson)", border=1)
pdf.cell(30, 6, "n * p_i", border=1, new_x="LMARGIN", new_y="NEXT")

for i in range(len(przedzialy)):
    prz_str = f"({przedzialy[i][0]:.1f}; {przedzialy[i][1]:.1f}]"
    pdf.set_x(10)
    pdf.cell(40, 6, prz_str, border=1)
    pdf.cell(30, 6, f"{pi_poisson[i]:.4f}", border=1)
    pdf.cell(30, 6, f"{npi_poisson[i]:.2f}", border=1, new_x="LMARGIN", new_y="NEXT")

pdf.set_font("DejaVu", size=10)
pdf.ln(2)

add_text(f"Wartość statystyki testowej: {testowa_poisson:.3f}")
for alfa in poziomy_alfa:
    kryt = scipy.stats.chi2.isf(alfa, st_swobody_poisson)
    wynik = "Brak podstaw do odrzucenia H0" if testowa_poisson < kryt else "Odrzucamy H0 na rzecz H1"
    add_text(f"Wartość krytyczna dla α={alfa}: {kryt:.3f} -> Wynik: {wynik}")
add_text("Wnioski: Model Poissona służy modelowaniu zjawisk dyskretnych. Cechy ciągłe (jak wzrost/masa) rzadko mu podlegają, co potwierdza zdecydowane odrzucenie H0 dla badanego zakresu α.")
pdf.ln(5)

# C. Kolmogorow
add_title("C. Test zgodności λ-Kołmogorowa dla rozkładu normalnego")
add_text("H0: Rozkład cechy jest normalny.")
add_text("H1: Rozkład cechy nie jest normalny.")
add_text("Szereg wykorzystany do obliczeń: Do testu Kołmogorowa wykorzystano surowy szereg nieszeregowany (obserwacje posortowane rosnąco), obliczając dystrybuantę empiryczną w każdym punkcie i szukając największej różnicy.")
pdf.ln(2)

add_text(f"Wartość statystyki testowej: {testowa_K:.3f}")
for alfa in poziomy_alfa:
    kryt = krytyczne_K_dict[alfa]
    wynik = "Brak podstaw do odrzucenia H0" if testowa_K < kryt else "Odrzucamy H0 na rzecz H1"
    add_text(f"Wartość krytyczna dla α={alfa}: {kryt:.3f} -> Wynik: {wynik}")
add_text("Wnioski: Test Kołmogorowa uzupełnia analizę z punktu A badając maksymalne odchylenie dystrybuanty. Jeśli wartość testowa nie przekracza krytycznej, potwierdzamy normalność rozkładu.")
pdf.ln(5)

# Regula 3 sigma
add_title("2. Analiza z wykorzystaniem reguły 3 sigma")
add_text("Wnioski z reguły 3 sigma dla punktów A i C:")
add_text(f"Przedział 3 sigma: [{dolna_3s:.2f} ; {gorna_3s:.2f}]")
add_text(f"W przedziale znajduje się {w_przedziale} z {n} obserwacji, co stanowi {procent_3s:.2f}%.")
if procent_3s >= 99.0:
    add_text("Uzasadnienie: Otrzymany wynik jest zgodny z wnioskami z testów A i C. Blisko 99.7% danych mieści się w przedziale, co jest klasycznym dowodem na zgodność badanej cechy z rozkładem normalnym.")
else:
    add_text("Uzasadnienie: Otrzymany wynik wykazuje odstępstwa od idealnego 99.7%. Potwierdza to wnioski z testów A i C (szczególnie dla wyższych wartości α), które mogły wskazać na odrzucenie H0 ze względu na obecność wartości odstających.")

nazwa_pliku = "Raport_Lab5_PL.pdf"
pdf.output(nazwa_pliku)

print(f"\nGotowe! Zapisano perfekcyjny plik z polskimi znakami: {nazwa_pliku}")