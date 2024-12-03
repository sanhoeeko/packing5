import math

import matplotlib
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.transforms import Affine2D


def getColorForInterval(color_map_name: str, interval: tuple):
    cmap = matplotlib.colormaps[color_map_name]
    a = interval[0]
    b = interval[1]
    k = 1 / (b - a)

    def callCmap(x: float):
        y = k * (x - a)
        return cmap(math.sqrt(y))

    return callCmap


def plotListOfArray(lst: list[np.ndarray]):
    cmap = getColorForInterval('cool', (0, len(lst)))
    for i in range(len(lst)):
        plt.plot(lst[i], color=cmap(i), alpha=0.5)
    plt.show()
    

def plotListOfArray3d(lst: list[np.ndarray]):
    n = max(map(len, lst))
    m = len(lst)
    mat = np.zeros((m, n))
    for i in range(len(lst)):
        mat[i, :len(lst[i])] = np.asarray(lst[i])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    xs = np.arange(0, n)
    ys = np.arange(0, m)
    X, Y = np.meshgrid(xs, ys)
    ax.plot_wireframe(X, Y, mat)
    plt.show()


class Capsule(patches.Patch):
    def __init__(self, xy, width, height, angle=0, **kwargs):
        """
        :param xy: The center of the capsule.
        :param width: The total width (diameter) of the capsule (including the line and semi-circles).
        :param height: The height (length) of the capsule.
        :param angle: The angle of rotation of the capsule.
        """
        self.xy = xy
        self.width = width
        self.height = height
        self.angle = angle
        super().__init__(**kwargs)

    def get_path(self):
        """
        Get the path of the capsule (a line with semi-circles at the ends).
        """
        # Create the path for the capsule
        c = self.width - self.height
        capsule_path = Path.make_compound_path(
            Path.unit_rectangle().transformed(
                Affine2D().scale(c, self.height).translate(-c / 2, -self.height / 2)),
            Path.unit_circle().transformed(Affine2D().scale(self.height / 2, self.height / 2).translate(-c / 2, 0)),
            Path.unit_circle().transformed(Affine2D().scale(self.height / 2, self.height / 2).translate(c / 2, 0))
        )

        return capsule_path

    def get_patch_transform(self):
        """
        Get the transformation for the capsule.
        """
        # Scale and rotate the capsule
        scale = Affine2D().scale(self.width, self.width).rotate_deg(self.angle)

        # Translate the capsule to the correct location
        trans = Affine2D().translate(*self.xy)

        return scale + trans
