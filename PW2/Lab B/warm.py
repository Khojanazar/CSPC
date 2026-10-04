import numpy as np
from scipy.optimize import newton, minimize

def grad_descent(df, x0, lr, tol=1e-8, max_iter=10000):
    x = x0
    for _ in range(max_iter):
        step = lr * df(x)
        x = x - step
        if abs(step) < tol:
            break
    return x

# ---------- 2A ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

print("2A: start x0 = 0")
x_gd = grad_descent(df, 0, lr=0.1)
x_nw = newton(df, 0, fprime=d2f)
x_sl = minimize(lambda v: f(v[0]), [0], method="SLSQP").x[0]
print(f"  gradient descent: x = {x_gd:.5f}")
print(f"  Newton:           x = {x_nw:.5f}")
print(f"  SLSQP:            x = {x_sl:.5f}")

# ---------- 2B ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

for x0 in (0, 2):
    print(f"\n2B: start x0 = {x0}")
    xg = grad_descent(dg, x0, lr=0.01)
    xn = newton(dg, x0, fprime=d2g)
    xs = minimize(lambda v: g(v[0]), [x0], method="SLSQP").x[0]
    kind = "MINIMUM" if d2g(xn) > 0 else "MAXIMUM"
    print(f"  gradient descent: x = {xg:.5f}, g = {g(xg):.5f}")
    print(f"  Newton:           x = {xn:.5f}, g = {g(xn):.5f}, g'' = {d2g(xn):.3f} -> {kind}")
    print(f"  SLSQP:            x = {xs:.5f}, g = {g(xs):.5f}")
