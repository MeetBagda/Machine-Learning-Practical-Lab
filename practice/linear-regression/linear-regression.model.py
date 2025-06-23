# m = (n * Σ(x*y) - Σx * Σy) / (n * Σ(x^2) - (Σx)^2)
# b = (Σy - m * Σx) / n

x = [1, 2, 3, 4, 5]
y = [3, 4, 2, 5, 6]

# Number of points
n = len(x)

sum_x = sum(x)
sum_y = sum(y)
sum_x2 = sum(i**2 for i in x)
sum_xy = sum(x[i]*y[i] for i in range(n))

# slope (m) and intercept (b)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
b = (sum_y - m * sum_x) / n

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")


def predict(x_val):
    return m * x_val + b


for xi in x:
    print(f"x={xi} => y={predict(xi):.2f}")
