import numpy as np
import statistics as stat

dane_szablon = []
with open("dane_wys_lab2.txt", 'r') as fi:
    for line in fi:
        if line.split():
            line = [float(x) for x in line.split()]
            dane_szablon.append(line)

dane=[]
for item in dane_szablon:
    for number in item:
        dane.append(number)

print(f'dane: {dane}')

# -------------------------------------------------------------------

min = np.min(dane)
max = np.max(dane)
n = len(dane)
rozstep = max-min
k_rob = round(n**0.5, 1)
print("")
print(f'min, max: {min, max, n , k_rob}')

# -------------------------------------------------------------------

def tworz_szereg(dane,  h, x01):
    max = np.max(dane)
    x11 = x01
    przedzialy = []
    liczebnosci = []
    zakresy = [x01]
    srodki = []
    while x11 < max:
        x11 = x01 + h
        n1 = len([x for x in dane if (x > x01 and x <= x11)])
        przedzialy.append([x01, x11])
        liczebnosci.append(n1)
        zakresy.append(x11)
        srodki.append((x01 + x11)/2)
        x01 = x11
    return przedzialy, liczebnosci, zakresy, srodki

print("")
print(tworz_szereg)
# -------------------------------------------------------------------

