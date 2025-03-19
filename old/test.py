import numpy as np
r = [0.3, 0.26666]
length = 9.14
density = 7860

# 6 elements of radius r[0], 4 of radius r[1], approximate lengths
# 6 of length=9.14, 4 of length=9.14*sqrt(2)
weight = (6 * np.pi * r[0]**2 * length + 4 * np.pi * r[1]**2 * length * np.sqrt(2)) * density
print(weight)