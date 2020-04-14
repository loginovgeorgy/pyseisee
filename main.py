from src import *
import numpy as np
import pylab as plt
from matplotlib import interactive
ns, nr = 1500, 200
x = np.linspace(0, 2*np.pi, ns)
traces = np.tile(np.sin(x*10), (nr, 1))

mg = PGather(traces, offset_ticks_freq=10)


# mg.update(gain=2)

plt.show()
