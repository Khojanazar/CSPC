import time
from decay import simulate, simulate_loop

N0 = 200000      
lam = 0.4


start = time.perf_counter()
simulate_loop(N0, lam)
end = time.perf_counter()
loop_time = end - start


start = time.perf_counter()
simulate(N0, lam)
end = time.perf_counter()
numpy_time = end - start

print(f"Pure Python loop: {loop_time:.4f} s")
print(f"NumPy vectorized: {numpy_time:.4f} s")
print(f"NumPy is {loop_time / numpy_time:.1f}x faster")