import numpy as np

def util_min(vector, **kwargs):
    return np.min(vector)


def util_mean(vector, **kwargs):
    return np.mean(vector)


def util_weighted_mean(vector, weights, **kwargs):
    return np.dot(vector, weights)


def util_hurwicz(vector, alpha, **kwargs):
    return alpha * np.max(vector) + (1 - alpha) * np.min(vector)


def find_best_by_utility(data, utility_func, **kwargs):
    utilities = [utility_func(vector, **kwargs) for vector in data]

    max_utility = max(utilities)

    return [i for i, u in enumerate(utilities) if np.isclose(u, max_utility)]


if __name__ == "__main__":
    print("\n--- ZADANIE 4 ---")
    data_z4 = np.array([
        [2, 8, 4],  # 0
        [5, 5, 5],  # 1
        [1, 9, 9]  # 2
    ])

    best_min = find_best_by_utility(data_z4, util_min)
    print(f"Najlepsze wg Min: {best_min}")

    best_mean = find_best_by_utility(data_z4, util_mean)
    print(f"Najlepsze wg Średniej: {best_mean}")

    wagi = [0.5, 0.3, 0.2]
    best_weighted = find_best_by_utility(data_z4, util_weighted_mean, weights=wagi)
    print(f"Najlepsze wg Średniej Ważonej (wagi={wagi}): {best_weighted}")

    wspolczynnik_ostroznosci = 0.6  # Współczynnik optymizmu (alfa)
    best_hurwicz = find_best_by_utility(data_z4, util_hurwicz, alpha=wspolczynnik_ostroznosci)
    print(f"Najlepsze wg kryterium Hurwicza (alfa={wspolczynnik_ostroznosci}): {best_hurwicz}")