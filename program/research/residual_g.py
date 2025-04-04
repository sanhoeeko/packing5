import numpy as np
import matplotlib.pyplot as plt

from analysis.database import Database

db = Database('../data-20250403.h5')
lst = [e.property('max_gradient_amp') for e in db]
pool = np.hstack(list(map(lambda x:x.reshape(-1), lst)))
log_pool = np.log10(pool)
plt.hist(log_pool, bins=50)
plt.show()
