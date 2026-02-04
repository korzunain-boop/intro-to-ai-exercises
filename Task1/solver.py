from abc import ABC, abstractmethod
import numpy as np
from calc_path import calc_target
from ga_functions import roulette_selection, one_point_crossover, mutate
from datetime import datetime
import matplotlib.pyplot as plt
class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(problem, pop, *args, **kwargs):
        start_time1 = datetime.now()
        start_time = float(start_time1.timestamp())
        tmax = kwargs.get("tmax", 200)
        mu = kwargs.get("mu", 10)
        pc = kwargs.get("pc", 0.8)
        pm = kwargs.get("pm", 0.02)  
        t = 0
        while True:
            values = [problem(chromosome) for chromosome in pop]
            max_value = values[0]
            for j in range(1, len(values)):
                if values[j] > max_value:
                    max_value = values[j]
                    jmax = j

            if max_value > -4:
                finish_time1 = datetime.now()
                finish_time = float(finish_time1.timestamp())
                return 0, pop, finish_time - start_time
            
            selected_indicies, elite_indicies = roulette_selection(values, 0.3, mu)
            elite = [pop[i] for i in elite_indicies]
            pop_after_selection = [elite[i] for i in selected_indicies]

            
            pop_after_crossover = pop_after_selection
            for i in range(mu // 2):
                if np.random.rand() < pc:
                    child1, child2 = one_point_crossover(pop_after_selection[2*i], pop_after_selection[2*i+1])
                    pop_after_selection[2*i] = child1
                    pop_after_selection[2*i+1] = child2
            [mutate(chromosome, pm) for chromosome in pop_after_crossover]

            pop_after_mutation = pop_after_crossover
                
            pop = pop_after_mutation
            t += 1
            if t > tmax+1:
                finish_time = datetime.now()
                finish_time = float(finish_time.timestamp())
                return -1, jmax, finish_time - start_time 

