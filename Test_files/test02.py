import numpy as np

a = ["1", "2", "3", "4", "1", "2", "3", "4"]
print(a, type(a[0]))
b = [int(x) for x in a]
print(b, type(b[0]))
c = list(map(int, a))
print(c, type(c[0]))

d = np.split(np.array(a), len(np.array(a)) / 4)
for e in d:
    print(e)
