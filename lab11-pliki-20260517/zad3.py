import numpy as np

def lexicographic_sort(data, hierarchy):
    n = len(data)

    vectors_with_indices = []
    for i in range(n):
        ordered_vector = tuple(data[i][h] for h in hierarchy)
        vectors_with_indices.append((ordered_vector, i))

    vectors_with_indices.sort(key=lambda x: x[0], reverse=True)

    return [idx for val, idx in vectors_with_indices]


if __name__ == "__main__":
    print("\n--- ZADANIE 3 ---")
    data_z3 = np.array([
        [2, 9, 3],  # Wektor 0
        [2, 8, 5],  # Wektor 1
        [4, 1, 1],  # Wektor 2
        [2, 9, 4]  # Wektor 3
    ])
    # Zakładamy, że cele są w kolejności: kolumna 0 (najważniejsza), kolumna 2, kolumna 1
    hierarchia = [0, 2, 1]

    lex_sorted = lexicographic_sort(data_z3, hierarchia)
    print(f"Wektory od najlepszego do najgorszego (hierarchia celów {hierarchia}): {lex_sorted}")