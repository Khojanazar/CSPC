"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize

def grad_descent(df, x0, lr, tol=1e-8, max_iter=10000):
    x = x0
    for _ in range(max_iter):
        step = lr * df(x)
        x = x - step
        if abs(step) < tol:      # stop when the step is tiny
            break
    return x

# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0


# TODO 2A: minimise f three ways from x0=0 and print each result:
#   (1) gradient descent by hand (loop x = x - lr*df(x) until the step is tiny)
#   (2) scipy.optimize.newton(df, x0, fprime=d2f)
#   (3) scipy.optimize.minimize(f, x0, method="SLSQP")


x_gd = grad_descent(df, 0, lr=0.1)
x_nw = newton(df, 0, fprime=d2f)
x_sl = minimize(lambda v: f(v[0]), [0], method="SLSQP").x[0]
print(f"  gradient descent: x = {x_gd:.5f}")
print(f" Newton:            x = {x_nw:.5f}")
print(f"  SLSQP:             x = {x_sl:.5f}")


# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

# TODO 2B: run the same three methods on g, from x0=0 AND from x0=2.
#   For Newton (which solves dg(x)=0), also check the sign of d2g at the answer:
#   d2g > 0 means a minimum, d2g < 0 means a maximum.
#   In your README note: do the methods agree? did Newton find a minimum or
#   another stationary point? how did the starting point change the result?

for x0 in (0, 2):

    xg = grad_descent(dg, x0, lr=0.01)
    xn = newton(dg, x0, fprime=d2g)
    xs = minimize(lambda v: g(v[0]), [x0], method="SLSQP").x[0]

    kind = "МИНИМУМ" if d2g(xn) > 0 else "МАКСИМУМ"
    print(f"  gradient descent: x = {xg:.5f}, g = {g(xg):.5f}")
    print(f"  Newton:            x = {xn:.5f}, g = {g(xn):.5f}, "
          f"g'' = {d2g(xn):.3f} -> {kind}")
    print(f"  SLSQP:             x = {xs:.5f}, g = {g(xs):.5f}")