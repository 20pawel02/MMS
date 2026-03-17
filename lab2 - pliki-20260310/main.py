import numpy as np
import matplotlib.pyplot as plt

# 1. Wczytywanie danych
dane = []
with open("dane_wys_lab2.txt", 'r') as fi:
    for line in fi:
        if line.strip():
            dane.extend([float(x) for x in line.split()])

dane = np.array(dane)
n = len(dane)

# 2. Statystyki z próby nieuporządkowanej (surowej) używając wyłącznie numpy
mean_raw = np.mean(dane)
std_raw = np.std(dane) # odch. std populacyjne
median_raw = np.median(dane)

# Wyznaczanie dominanty (mody) z danych surowych bez użycia scipy
vals, counts = np.unique(dane, return_counts=True)
mode_raw = vals[np.argmax(counts)]

cv_raw = std_raw / mean_raw * 100

# Wyznaczanie współczynnika asymetrii (skośności) bez użycia scipy (Trzeci moment centralny / s^3)
m3_raw = np.sum((dane - mean_raw)**3) / n
skew_raw = m3_raw / (std_raw**3)

mix_skew_raw = 3 * (mean_raw - median_raw) / std_raw

# Wartości typowe i odstające
t1_l, t1_p = mean_raw - std_raw, mean_raw + std_raw
p_typowe1 = len([x for x in dane if t1_l <= x <= t1_p]) / n * 100

t2_l, t2_p = mean_raw - 2*std_raw, mean_raw + 2*std_raw
p_typowe2 = len([x for x in dane if t2_l <= x <= t2_p]) / n * 100

outliers_left = len([x for x in dane if x < t2_l])
outliers_right = len([x for x in dane if x > t2_p])


# 3. Funkcja do wyznaczania statystyk dla szeregu rozdzielczego
def calc_grouped_stats(dane, h, x01):
    max_val = np.max(dane)
    x11 = x01
    przedzialy, liczebnosci, srodki = [], [], []
    
    # Tworzenie szeregu
    while x11 < max_val:
        x11 = x01 + h
        n1 = len([x for x in dane if x > x01 and x <= x11])
        przedzialy.append((x01, x11))
        liczebnosci.append(n1)
        srodki.append((x01 + x11) / 2)
        x01 = x11
    
    n_g = sum(liczebnosci)
    mean_g = sum(m * f for m, f in zip(srodki, liczebnosci)) / n_g
    std_g = np.sqrt(sum(f * (m - mean_g)**2 for m, f in zip(srodki, liczebnosci)) / n_g)
    
    # Mediana z szeregu
    cum_f = np.cumsum(liczebnosci)
    med_class_idx = np.where(cum_f >= n_g / 2)[0][0]
    f_m = liczebnosci[med_class_idx]
    cf_prev = cum_f[med_class_idx - 1] if med_class_idx > 0 else 0
    L_med = przedzialy[med_class_idx][0]
    median_g = L_med + ((n_g / 2 - cf_prev) / f_m) * h
    
    # Dominanta z szeregu
    mod_class_idx = np.argmax(liczebnosci)
    f_mod = liczebnosci[mod_class_idx]
    f_mod_prev = liczebnosci[mod_class_idx - 1] if mod_class_idx > 0 else 0
    f_mod_next = liczebnosci[mod_class_idx + 1] if mod_class_idx < len(liczebnosci) - 1 else 0
    L_mod = przedzialy[mod_class_idx][0]
    
    if (f_mod - f_mod_prev) + (f_mod - f_mod_next) == 0:
        mode_g = L_mod  # Zabezpieczenie przed dzieleniem przez zero
    else:
        mode_g = L_mod + ((f_mod - f_mod_prev) / ((f_mod - f_mod_prev) + (f_mod - f_mod_next))) * h
    
    # Miary rozproszenia i asymetrii z szeregu
    cv_g = std_g / mean_g * 100
    m3_g = sum(f * (m - mean_g)**3 for m, f in zip(srodki, liczebnosci)) / n_g
    skew_g = m3_g / (std_g**3) if std_g > 0 else 0
    mix_skew_g = 3 * (mean_g - median_g) / std_g if std_g > 0 else 0
    
    return {
        'mean': mean_g, 'std': std_g, 'median': median_g, 'mode': mode_g,
        'cv': cv_g, 'skew': skew_g, 'mix_skew': mix_skew_g
    }

def format_err(g_val, raw_val):
    if raw_val == 0: return 0
    return abs(g_val - raw_val) / abs(raw_val) * 100

# Trzy warianty szeregów
g1 = calc_grouped_stats(dane, h=3.5, x01=128)  # Najlepszy wg roboczego wyliczenia
g2 = calc_grouped_stats(dane, h=4.5, x01=128)  # Zmienione h
g3 = calc_grouped_stats(dane, h=3.5, x01=125)  # Zmienione x01

# --- Wypisywanie wyników sformatowanych jako tabela ---

print(f"{'Cecha':<45} | {'Próba nieuporz.':<15} | {'Szereg najlep.':<14} | {'Zmienione h':<14} | {'Zmienione x01':<13}")
print("-" * 110)

def print_row(name, v_raw, v_g1, v_g2, v_g3, is_err=False, is_x=False):
    if is_x:
        print(f"{name:<45} | {v_raw:<15.2f} | {'x':<14} | {'x':<14} | {'x':<13}")
    elif is_err:
        print(f"{name:<45} | {'-':<15} | {v_g1:<14.2f} | {v_g2:<14.2f} | {v_g3:<13.2f}")
    else:
        print(f"{name:<45} | {v_raw:<15.2f} | {v_g1:<14.2f} | {v_g2:<14.2f} | {v_g3:<13.2f}")

print_row('średnia', mean_raw, g1['mean'], g2['mean'], g3['mean'])
print_row('błąd średniej (%)', 0, format_err(g1['mean'], mean_raw), format_err(g2['mean'], mean_raw), format_err(g3['mean'], mean_raw), is_err=True)
print_row('odch. Standard.', std_raw, g1['std'], g2['std'], g3['std'])
print_row('błąd odchylenia (%)', 0, format_err(g1['std'], std_raw), format_err(g2['std'], std_raw), format_err(g3['std'], std_raw), is_err=True)
print_row('mediana', median_raw, g1['median'], g2['median'], g3['median'])
print_row('błąd mediany (%)', 0, format_err(g1['median'], median_raw), format_err(g2['median'], median_raw), format_err(g3['median'], median_raw), is_err=True)
print_row('dominanta', mode_raw, g1['mode'], g2['mode'], g3['mode'])
print_row('błąd dominanty (%)', 0, format_err(g1['mode'], mode_raw), format_err(g2['mode'], mode_raw), format_err(g3['mode'], mode_raw), is_err=True)
print_row('wsp. zmienności', cv_raw, g1['cv'], g2['cv'], g3['cv'])
print_row('wsp. asymetrii', skew_raw, g1['skew'], g2['skew'], g3['skew'])
print_row('mieszany wsp.asymetrii', mix_skew_raw, g1['mix_skew'], g2['mix_skew'], g3['mix_skew'])
print_row('procent typowe1', p_typowe1, 0, 0, 0, is_x=True)
print_row('procent typowe2', p_typowe2, 0, 0, 0, is_x=True)
print_row('liczba wartości odstających po prawej stronie', outliers_right, 0, 0, 0, is_x=True)
print_row('liczba wartości odstających po lewej stronie', outliers_left, 0, 0, 0, is_x=True)