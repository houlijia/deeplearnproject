import numpy as np
import torch


print(np.random.seed(0))

N, D = 3, 4

x = np.random.randn(N, D)
print(f"x = {x}")
y = np.random.randn(N, D)
print(f"y = {y}")
z = np.random.randn(N, D)
print(f"z= {z}")

a = x * y
print(f"a = {a}")
b = a + z
print(f"b = {b}")
c = np.sum(b)
print(f"c = {c}")


grad_c = 1.0
grad_b = grad_c * np.ones((N, D))
print(f"grad_b = {grad_b}")
grad_a = grad_b.copy()
print(f"grad_a = {grad_a}")
grad_z = grad_b.copy()
print(f"grad_z = {grad_z}")
grad_x = grad_a * y
print(f"grad_x = {grad_x}")
grad_y = grad_a * x
print(f"grad_y = {grad_y}")