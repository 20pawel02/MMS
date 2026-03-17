import numpy as np

dane_skocz = []
with open("dane.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    header = lines[0].strip().split(',')
    idx_wiek = header.index('Wiek')
    idx_skocz = header.index('Skocz')
    
    for line in lines[1:]:
        if not line.strip(): continue
        parts = line.strip().split(',')
        try:
            wiek = float(parts[idx_wiek])
            if 12.0 <= wiek < 13.0:
                dane_skocz.append(float(parts[idx_skocz]))
        except ValueError:
            pass

dane = np.array(dane_skocz)
n = len(dane)

# kolumna: proba nieuporzadkowana

mean_raw = np.mean(dane)                 # średnia
std_raw = np.std(dane)                   # odch. standardowe
median_raw = np.median(dane)             # mediana

vals, counts = np.unique(dane, return_counts=True)
mode_raw = vals[np.argmax(counts)]       # dominanta

cv_raw = std_raw / mean_raw * 100        # współczynnik zmienności

m3_raw = np.sum((dane - mean_raw)**3) / n
skew_raw = m3_raw / (std_raw**3)         # współczynnik asymetrii

mix_skew_raw = 3 * (mean_raw - median_raw) / std_raw  # mieszany wsp. asymetrii

# procent wartości typowych ±1σ
t1_l, t1_p = mean_raw - std_raw, mean_raw + std_raw
p_typowe1 = len([x for x in dane if t1_l <= x <= t1_p]) / n * 100

# procent wartości typowych ±2σ
t2_l, t2_p = mean_raw - 2*std_raw, mean_raw + 2*std_raw
p_typowe2 = len([x for x in dane if t2_l <= x <= t2_p]) / n * 100

# liczba wartości odstających
outliers_left = len([x for x in dane if x < t2_l])
outliers_right = len([x for x in dane if x > t2_p])

# KOLUMNA: "szereg najlepszy"
# (miary liczone z szeregu rozdzielczego)
# 

h = 3
x01 = 11.5

max_val = np.max(dane)
x11 = x01
przedzialy, liczebnosci, srodki = [], [], []

while x11 < max_val:
    x11 = x01 + h
    n1 = len([x for x in dane if x > x01 and x <= x11])
    przedzialy.append((x01, x11))
    liczebnosci.append(n1)
    srodki.append((x01 + x11) / 2)
    x01 = x11

n_g = sum(liczebnosci)

mean_g = sum(m * f for m, f in zip(srodki, liczebnosci)) / n_g   # średnia z szeregu

std_g = np.sqrt(sum(f * (m - mean_g)**2 for m, f in zip(srodki, liczebnosci)) / n_g)  # odch std z szeregu

cum_f = np.cumsum(liczebnosci)
med_class_idx = np.where(cum_f >= n_g / 2)[0][0]
f_m = liczebnosci[med_class_idx]
cf_prev = cum_f[med_class_idx - 1] if med_class_idx > 0 else 0
L_med = przedzialy[med_class_idx][0]

median_g = L_med + ((n_g / 2 - cf_prev) / f_m) * h   # mediana z szeregu

mod_class_idx = np.argmax(liczebnosci)
f_mod = liczebnosci[mod_class_idx]
f_mod_prev = liczebnosci[mod_class_idx - 1] if mod_class_idx > 0 else 0
f_mod_next = liczebnosci[mod_class_idx + 1] if mod_class_idx < len(liczebnosci) - 1 else 0
L_mod = przedzialy[mod_class_idx][0]

mode_g = L_mod + ((f_mod - f_mod_prev) / ((f_mod - f_mod_prev) + (f_mod - f_mod_next))) * h  # dominanta z szeregu

cv_g = std_g / mean_g * 100                      # współczynnik zmienności z szeregu

m3_g = sum(f * (m - mean_g)**3 for m, f in zip(srodki, liczebnosci)) / n_g
skew_g = m3_g / (std_g**3) if std_g > 0 else 0   # asymetria z szeregu

mix_skew_g = 3 * (mean_g - median_g) / std_g if std_g > 0 else 0  # mieszana asymetria z szeregu

# kolumna "bledy"

def format_err(g_val, raw_val):
    if raw_val == 0: return 0
    return abs(g_val - raw_val) / abs(raw_val) * 100

# KOLUMNA: "klasyfikacja Lundmana"
# (liczona TYLKO z danych surowych Rohrera)

lepto = np.sum(dane < 135)
atlet = np.sum((dane >= 135) & (dane < 145))
pykni = np.sum((dane >= 145) & (dane < 155))
hiper = np.sum(dane >= 155)

print("\n--- Klasyfikacja Lundmana ---")
print("Leptosomatyczny:", lepto)
print("Atletyczny:", atlet)
print("Pykniczny:", pykni)
print("Hiperpykniczny:", hiper)

# wyniki

print("\n------------------------------")
print(f"Średnia surowa: {mean_raw:.2f}, z szeregu: {mean_g:.2f} (błąd: {format_err(mean_g, mean_raw):.2f}%)")
print(f"Błędy odchylenia: {format_err(std_g, std_raw):.2f}%")

print("\n===== PRÓBA NIEUPORZĄDKOWANA =====")
print("Średnia:", round(mean_raw,2))
print("Odch std:", round(std_raw,2))
print("Mediana:", round(median_raw,2))
print("Dominanta:", round(mode_raw,2))
print("Wsp zmienności:", round(cv_raw,2))
print("Asymetria:", round(skew_raw,2))
print("Mieszana asymetria:", round(mix_skew_raw,2))
print("% typowe1:", round(p_typowe1,2))
print("% typowe2:", round(p_typowe2,2))
print("Odstające lewe:", outliers_left)
print("Odstające prawe:", outliers_right)

print("\n===== SZEREG NAJLEPSZY =====")
print("Średnia:", round(mean_g,2))
print("Odch std:", round(std_g,2))
print("Mediana:", round(median_g,2))
print("Dominanta:", round(mode_g,2))
print("Wsp zmienności:", round(cv_g,2))
print("Asymetria:", round(skew_g,2))
print("Mieszana asymetria:", round(mix_skew_g,2))

print("\n===== BŁĘDY =====")
print("Błąd średniej:", round(format_err(mean_g, mean_raw),2))
print("Błąd odchylenia:", round(format_err(std_g, std_raw),2))
print("Błąd mediany:", round(format_err(median_g, median_raw),2))
print("Błąd dominanty:", round(format_err(mode_g, mode_raw),2))

print("\n===== KLASYFIKACJA LUNDMANA =====")
print("Lepto:", lepto)
print("Atlet:", atlet)
print("Pykniczny:", pykni)
print("Hiper:", hiper)