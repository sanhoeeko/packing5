import math

import matplotlib
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.transforms import Affine2D


class Figure:
    def __enter__(self):
        self.fig, self.ax = plt.subplots()
        return self.ax

    def __exit__(self, exc_type, exc_value, traceback):
        plt.show()


def getColorForInterval(color_map_name: str, interval: tuple):
    cmap = matplotlib.colormaps[color_map_name]
    a = interval[0]
    b = interval[1]
    k = 1 / (b - a)

    def callCmap(x: float):
        y = k * (x - a)
        return cmap(math.sqrt(y))

    return callCmap


def plotListOfArray(lst: np.ndarray):
    cmap = getColorForInterval('cool', (0, len(lst)))
    with Figure() as ax:
        for i in range(len(lst)):
            ax.plot(lst[i], color=cmap(i), alpha=0.5)


def plotMeanCurvesWithCI(x: np.ndarray, y_mean_lst: list[np.ndarray], y_ci_lst: list[np.ndarray]):
    cmap_s = 'cool'
    cmap = plt.get_cmap(cmap_s)
    assert len(y_mean_lst) == len(y_ci_lst)
    colors = cmap(np.linspace(0, 1, len(y_mean_lst)))
    with Figure() as ax:
        for i, (y_mean, y_ci) in enumerate(zip(y_mean_lst, y_ci_lst)):
            color = colors[i]
            ax.fill_between(x, y_mean - y_ci, y_mean + y_ci, color=color, alpha=0.2)
            ax.plot(x, y_mean, color=color)

    # if self.average_flags is not None and len(self.average_flags) == 1:
    #     flag = self.average_flags[0]
    #     lst = np.asarray(self.abstract[flag])
    #     min_v, max_v = np.min(lst), np.max(lst)
    #     norm = Normalize(vmin=min_v, vmax=max_v)
    #     sm = ScalarMappable(cmap=cmap_s, norm=norm)
    #     cbar = plt.colorbar(sm)
    #     cbar.set_label(flag)
    # if self.x_label is not None:
    #     plt.xlabel(self.x_label)
    # if self.y_label is not None:
    #     plt.ylabel(self.y_label)


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
