import numpy as np
from solver import Solver
from calc_path import calc_target
import matplotlib.pyplot as plt

# To begin tests uncomment 1 or many section that you need. You can change repeats parametr to lower the error
# repeats = 5 # testing mu impact on function execution time
# times = np.zeros(19)
# mus = [2*i for i in range(2, 21)]
# for j in range(1, repeats):
#     pop = [np.random.randint(0, 2, size=(2000, 2)) for _ in range(100)]
#     for i in range(2, 21):
#         a, b, func_time = Solver.solve(calc_target, pop, mu=i*2)
#         times[i-2] += func_time/repeats
# plt.scatter(mus, times)
# plt.xlabel("mu")
# plt.ylabel("time")
# plt.savefig('mus_comparison.png')  # mu = 8
# plt.show()


# repeats = 5  # testing pc impact on function execution time
# times = np.zeros(24)
# pcs = [0.02*i + 0.5 for i in range(0, 24)]
# for j in range(1, repeats):
#     pop = [np.random.randint(0, 2, size=(2000, 2)) for _ in range(100)]
#     for i in range(1, 25):
#         a, b, func_time = Solver.solve(calc_target, pop, mu=8, pc=pcs[i-1])
#         times[i-1] += func_time/repeats
# plt.scatter(pcs, times)
# plt.xlabel("pc")
# plt.ylabel("time")
#plt.savefig('pc_comparison.png') 
# plt.show() # pc = 0.86


repeats = 5  # testing pm impact on function execution time
times = np.zeros(14)
pms = [0.01*i for i in range(1, 15)]
for j in range(1, repeats):
    pop = [np.random.randint(0, 2, size=(2000, 2)) for _ in range(100)]
    for i in range(1, 15):
        a, b, func_time = Solver.solve(calc_target, pop, mu=8, pc=0.86, pm=pms[i-1])
        times[i-1] += func_time/repeats
plt.scatter(pms, times)
plt.xlabel("pc")
plt.ylabel("time")
plt.savefig('pm_comparison.png') 
plt.show() # pm = 0.02
