from banana.interfaces.coils import arlo
from scipy.io import loadmat

y = loadmat('/Users/tclose/Desktop/y.txt')['y']
r2_ref = loadmat('/Users/tclose/Desktop/r2s.txt')['r2']

r2 = arlo([2, 5, 9], y)

print(r2_ref - r2)
print(r2)
