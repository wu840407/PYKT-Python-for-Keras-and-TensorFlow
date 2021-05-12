import matplotlib.pyplot as plt
import numpy as np

# y= ax+b
b = np.linspace(5, -5, 11)
print(b)
a = -3
x = np.arange(-10, 10, 0.1)
#print(x)

for b1 in b:
    y = a * x + b1
    plt.plot(x, y, label=f"y={a}x+{b1}")
    plt.legend(loc=2)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title("demo2 figure")
plt.xlabel("label for x")
plt.ylabel("label for y")
# plt.xticks([])
# plt.yticks([])
plt.show()