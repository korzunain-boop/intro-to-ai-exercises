import numpy as np
from datetime import datetime

def get_time():
    return None

def roulette_selection(values, fraction,  k: int):
    values = np.array(values, dtype=float)
    elite_size = max(1, int(len(values) * fraction))
    elite_indices = np.argsort(values)[-elite_size:]
    values = values[elite_indices]
    shifted = values - min(values)
    total = np.sum(shifted)
    probs = shifted/total
    if total == 0:
        probs = [1/len(values) for i in values]
    selected_indices = np.random.choice(len(values), size=k, p=probs)
    return selected_indices, elite_indices

def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray):

    point = np.random.randint(1, parent1.shape[0])  

    child1 = np.vstack([parent1[:point, :], parent2[point:, :]])
    child2 = np.vstack([parent2[:point, :], parent1[point:, :]])

    return child1, child2



def mutate(chromosome, pm):
    for j in [0, 1] :
        for i in range(len(chromosome)):
            if np.random.rand() < pm:
                chromosome[i][j] = 1 - chromosome[i][j]

