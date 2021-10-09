from add_McCormick_envelope_constraint import limit2vertex
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

bilinear_limits = np.array([0.0, 1.0, -1.0, 1.0])

v = limit2vertex(bilinear_limits)

fig = plt.figure()
ax = Axes3D(fig)

for iter_v in range(4):
    ax.scatter(v[iter_v, 0], v[iter_v, 1], v[iter_v, 2], marker='.')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
