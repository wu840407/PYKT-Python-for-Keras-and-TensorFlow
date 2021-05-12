import matplotlib.pyplot as plt
import numpy as np

label = 'w=%.1f'
label1 = 'w={:.1f}'
w1 = 0.5
w2 = 1.0
w3 = 2.0
# l1 = label % w1
# print(l1)
l1 = label1.format(w1)
l2 = label % w2
l3 = label % w3
x = np.arange(-10, 10, 0.1)
for w, l in zip([w1, w2, w3], [l1, l2, l3]):
    f = 1 / (1 + np.exp(-x * w + 0))
    plt.plot(x, f, label=l)
plt.legend(loc=2)
plt.show()