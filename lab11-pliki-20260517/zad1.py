import pandas as pd
import numpy as np

def dominates_pareto_max(a, b):
    return np.all(a >= b) and np.any(a > b)

def find_pareto_optimal_max(data):
    n = len(data)
    is_dominated = [False] * n
    
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            
            if dominates_pareto_max(data[i], data[j]):
                is_dominated[j] = True
            elif dominates_pareto_max(data[j], data[i]):
                is_dominated[i] = True
                break 
                
    return [i for i, dominated in enumerate(is_dominated) if not dominated]

if __name__ == "__main__":
    print("--- ZADANIE 1 ---")

    df1 = pd.read_excel('lab11-26.xlsx', header=None)
    data_z1 = df1.values

    # przykładowe dane
    data_z1 = np.array([
        [5, 5, 5], # 0
        [2, 4, 3], # 1 (zdominowany przez 0)
        [6, 3, 5], # 2 (optymalny)
        [1, 1, 1]  # 3 (zdominowany)
    ])
    
    opt_max = find_pareto_optimal_max(data_z1)
    print(f"Indeksy wektorów optymalnych (Pareto Max): {opt_max}")