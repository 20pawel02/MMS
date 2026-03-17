import numpy as np
import csv

# =========================================
# WCZYTANIE DANYCH – SKOCZNOŚĆ 12-LATKÓW
# =========================================

skocz = []

with open("dane.csv", newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        w = float(row["Wiek"])
        if 12.00 <= w <= 12.99:
            skocz.append(float(row["Skocz"]))

x = np.array(skocz)
n = len(x)

# =========================================
# PRÓBA NIEUPORZĄDKOWANA
# =========================================

mean_raw = np.mean(x)
std_raw = np.std(x, ddof=1)
median_raw = np.median(x)

vals, counts = np.unique(x, return_counts=True)
mode_raw = vals[np.argmax(counts)]

cv_raw = std_raw / mean_raw * 100

skew_raw = np.mean((x-mean_raw)**3) / (std_raw**3)
mix_skew_raw = 3*(mean_raw-median_raw)/std_raw

typ1 = np.sum(np.abs(x-mean_raw) <= std_raw)/n*100
typ2 = np.sum(np.abs(x-mean_raw) <= 2*std_raw)/n*100

out_r = np.sum(x > mean_raw + 2*std_raw)
out_l = np.sum(x < mean_raw - 2*std_raw)

# =========================================
# NAJLEPSZY SZEREG (STURGES)
# =========================================

k = int(np.round(1 + 3.322*np.log10(n)))

xmin = np.min(x)
xmax = np.max(x)

h = (xmax-xmin)/k

bins = np.arange(xmin, xmax+h, h)

freq, edges = np.histogram(x, bins=bins)
mid = (edges[:-1] + edges[1:])/2

# =========================================
# STATYSTYKI Z SZEREGU
# =========================================

mean_g = np.sum(mid*freq)/n

std_g = np.sqrt(np.sum(freq*(mid-mean_g)**2)/(n-1))

cum = np.cumsum(freq)
med_class = np.where(cum >= n/2)[0][0]

L = edges[med_class]
cf_prev = cum[med_class-1] if med_class>0 else 0
f = freq[med_class]

median_g = L + ((n/2-cf_prev)/f)*h

mod_class = np.argmax(freq)

Lmod = edges[mod_class]
f1 = freq[mod_class]
f0 = freq[mod_class-1] if mod_class>0 else 0
f2 = freq[mod_class+1] if mod_class<len(freq)-1 else 0

mode_g = Lmod + ((f1-f0)/((f1-f0)+(f1-f2)))*h

cv_g = std_g/mean_g*100

skew_g = np.sum(freq*(mid-mean_g)**3)/n / (std_g**3)
mix_skew_g = 3*(mean_g-median_g)/std_g

# =========================================
# BŁĘDY
# =========================================

def err(a,b):
    return abs(a-b)

err_mean = err(mean_g, mean_raw)
err_std = err(std_g, std_raw)
err_med = err(median_g, median_raw)
err_mod = err(mode_g, mode_raw)

# =========================================
# WYPISYWANIE DO TABELI Z ZAD3
# =========================================

print("\n===== PRÓBA NIEUPORZĄDKOWANA =====")
print("Średnia:", round(mean_raw,2))
print("Odch std:", round(std_raw,2))
print("Mediana:", round(median_raw,2))
print("Dominanta:", round(mode_raw,2))
print("Wsp zmienności:", round(cv_raw,2))
print("Asymetria:", round(skew_raw,2))
print("Mieszana asymetria:", round(mix_skew_raw,2))
print("% typowe1:", round(typ1,2))
print("% typowe2:", round(typ2,2))
print("Odstające prawe:", out_r)
print("Odstające lewe:", out_l)

print("\n===== SZEREG NAJLEPSZY =====")
print("Średnia:", round(mean_g,2))
print("Odch std:", round(std_g,2))
print("Mediana:", round(median_g,2))
print("Dominanta:", round(mode_g,2))
print("Wsp zmienności:", round(cv_g,2))
print("Asymetria:", round(skew_g,2))
print("Mieszana asymetria:", round(mix_skew_g,2))

print("\n===== BŁĘDY =====")
print("Błąd średniej:", round(err_mean,2))
print("Błąd odchylenia:", round(err_std,2))
print("Błąd mediany:", round(err_med,2))
print("Błąd dominanty:", round(err_mod,2))