import numpy as np
import matplotlib.pyplot as plt

# wczytywanie danych
dane_sz = []
with open("dane_wys_lab2.txt", 'r') as fi:
    for line in fi:
        if line.split():
            line = [float(x) for x in line.split()]
            dane_sz.append(line)


dane=[]
for item in dane_sz:
    for number in item:
        dane.append(number)

