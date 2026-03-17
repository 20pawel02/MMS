import numpy as np
import statistics as stat

dane_cale = []
with open("lab1_dane.txt", 'r') as fi:
    for line in fi:
        if line.split():
            line = [float(x) for x in line.split()]
            dane_cale.append(line)
print(dane_cale)


dane4 = []
dane1 = []
dane2 = []
dane3 = []
for item in dane_cale:
    dane1.append(item[0])
    dane2.append(item[1])
    dane3.append(item[2])
    dane4.append(item[3])
print(dane1)    


dane = dane1
x_sr = round(np.mean(dane), 2)
s = round(np.std(dane), 2) #odch standardowe populacji
s_p = round(np.std(dane, ddof=1), 2) #odch standardowe z próby
Q1 = round(np.percentile(dane, 25), 2)
M = round(np.percentile(dane, 50), 2)
Q3 = round(np.percentile(dane, 75), 2)
min = np.min(dane)
max = np.max(dane)
print("średnia", x_sr)
print("odch.standard populacji", s)
print("odch.standard z próby", s_p)
print("kwartyl1", Q1)
print("mediana", M)
print("kwartyl3", Q3)
print("min", min)
print("max", max)


# mean()   Arithmetic mean (“average”) of data.
# quantiles()  Divide data into intervals with equal probability.
# median()  Median (middle value) of data.
# pstdev()  Population standard deviation of data.
# stdev()  Sample standard deviation of data.
x_sr = round(stat.mean(dane), 2)
s_p = round(stat.stdev(dane), 2)
s = round(stat.pstdev(dane), 2)
kwartyle = stat.quantiles(dane, n = 4)
Q1 = kwartyle[0]
Q3 = kwartyle[2]
M = stat.median(dane)

print("średnia", x_sr)
print("odch.standard populacji", s)
print("odch.standard z próby", s_p)
print("kwartyle", kwartyle)
print("mediana", M)


t1_l = round(x_sr - s, 2)
t1_p = round(x_sr + s, 2)
typowe1 = [x for x in dane if (x >= t1_l and x <= t1_p)]
print(t1_l, t1_p,  typowe1)

t2_l = round(x_sr - 2*s, 2)
t2_p = round(x_sr + 2*s, 2)
typowe2 = [x for x in dane if (x >= t2_l and x <= t2_p)]
print(t2_l, t2_p,  typowe2)


odstajace = [x for x in dane if (x < t2_l or x > t2_p)]
print(odstajace)


#procent typowe1
p1 = round(len(typowe1)/len(dane), 3)
print(p1)


#procent typowe2
p2 = round(len(typowe2)/len(dane), 3)
print(p2)