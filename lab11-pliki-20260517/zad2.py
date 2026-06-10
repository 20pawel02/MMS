import numpy

def dominates_pareto_min(a, b):
    return np.all(a <= b) and np.any(a < b)

def find_pareto_optimal_min_cols(data_cols):
    data = np.transpose(data_cols)
    n = len(data)
    is_dominated = [False] * n
    
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if dominates_pareto_min(data[i], data[j]):
                is_dominated[j] = True
            elif dominates_pareto_min(data[j], data[i]):
                is_dominated[i] = True
                break
                
    return [i for i, dominated in enumerate(is_dominated) if not dominated]

if __name__ == "__main__":
    print("\n--- ZADANIE 2 ---")
    # df2 = pd.read_csv('lab11-26.xlsx - Zadanie2.csv', header=None)
    # data_z2 = df2.values
    
    # Przykładowe dane kolumnowe (kolumny to wektory k0, k1, k2)
    # k0: [10, 10], k1: [5, 12], k2: [12, 12]
    data_z2 = np.array([
        [10, 5, 12], 
        [10, 12, 12]
    ])
    
    opt_min = find_pareto_optimal_min_cols(data_z2)
    print(f"Indeksy wektorów optymalnych (Pareto Min, kolumny): {opt_min}")