import numpy as np
sizes = [2, 3, 1]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
nabla_w = [np.zeros(w.shape) for w in weights]
print weights
print nabla_w