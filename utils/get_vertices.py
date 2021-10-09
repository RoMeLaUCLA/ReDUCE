import numpy as np
import matplotlib.pyplot as plt


def get_vertices(center, R_bw, size):
    assert len(center) == 2, "Length of center is not 2!"
    assert len(size) == 2, "Length of size vector is not 2!"
    assert R_bw.shape == (2, 2), "Incorrect shape for rotation matrix!"

    Height = size[0]
    Width = size[1]

    v = np.zeros([4, 2])
    v[0, :] = center + R_bw.dot(np.array([Width / 2.0, Height / 2.0]))
    v[1, :] = center + R_bw.dot(np.array([Width / 2.0, -Height / 2.0]))
    v[2, :] = center + R_bw.dot(np.array([-Width / 2.0, -Height / 2.0]))
    v[3, :] = center + R_bw.dot(np.array([-Width / 2.0, Height / 2.0]))

    return v


def plot_rectangle(ax, v, color, show):
    ax.plot([v[0, 0], v[1, 0]], [v[0, 1], v[1, 1]], color=color)
    ax.plot([v[1, 0], v[2, 0]], [v[1, 1], v[2, 1]], color=color)
    ax.plot([v[2, 0], v[3, 0]], [v[2, 1], v[3, 1]], color=color)
    ax.plot([v[3, 0], v[0, 0]], [v[3, 1], v[0, 1]], color=color)

    if show:
        plt.show()


def plot_bilinear(v_all, num_of_vertices, num_of_polygons):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)

    for iter_v in range(num_of_vertices):
        ax.scatter(v_all[0, iter_v], v_all[1, iter_v], v_all[2, iter_v], marker='.')

    for iter_polygon in range(num_of_polygons):
        ax.plot3D([v_all[0, 4 * iter_polygon + 0], v_all[0, 4 * iter_polygon + 1]],
                  [v_all[1, 4 * iter_polygon + 0], v_all[1, 4 * iter_polygon + 1]],
                  [v_all[2, 4 * iter_polygon + 0], v_all[2, 4 * iter_polygon + 1]], 'black')

        ax.plot3D([v_all[0, 4 * iter_polygon + 0], v_all[0, 4 * iter_polygon + 2]],
                  [v_all[1, 4 * iter_polygon + 0], v_all[1, 4 * iter_polygon + 2]],
                  [v_all[2, 4 * iter_polygon + 0], v_all[2, 4 * iter_polygon + 2]], 'black')

        ax.plot3D([v_all[0, 4 * iter_polygon + 0], v_all[0, 4 * iter_polygon + 3]],
                  [v_all[1, 4 * iter_polygon + 0], v_all[1, 4 * iter_polygon + 3]],
                  [v_all[2, 4 * iter_polygon + 0], v_all[2, 4 * iter_polygon + 3]], 'black')

        ax.plot3D([v_all[0, 4 * iter_polygon + 1], v_all[0, 4 * iter_polygon + 3]],
                  [v_all[1, 4 * iter_polygon + 1], v_all[1, 4 * iter_polygon + 3]],
                  [v_all[2, 4 * iter_polygon + 1], v_all[2, 4 * iter_polygon + 3]], 'black')

        ax.plot3D([v_all[0, 4 * iter_polygon + 2], v_all[0, 4 * iter_polygon + 3]],
                  [v_all[1, 4 * iter_polygon + 2], v_all[1, 4 * iter_polygon + 3]],
                  [v_all[2, 4 * iter_polygon + 2], v_all[2, 4 * iter_polygon + 3]], 'black')

        ax.plot3D([v_all[0, 4 * iter_polygon + 1], v_all[0, 4 * iter_polygon + 2]],
                  [v_all[1, 4 * iter_polygon + 1], v_all[1, 4 * iter_polygon + 2]],
                  [v_all[2, 4 * iter_polygon + 1], v_all[2, 4 * iter_polygon + 2]], 'black')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
