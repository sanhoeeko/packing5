import matplotlib
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

my_colors = ['floralWhite', 'lemonchiffon', 'wheat', 'lightsalmon', 'coral', 'crimson',
             'paleturquoise', 'blue', 'teal', 'seagreen', 'green']

matplotlib.rcParams.update({
    'font.size': 22
})


class Figure:
    """
    It can be created as a normal object, or at the beginning of a `with` block
    """

    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        plt.show()

    def clear(self):
        self.ax.clear()
        exclude_targets = [self.ax_slider] if hasattr(self, 'ax_slider') else []
        for c in self.fig.get_axes():
            if c is not self.ax and c not in exclude_targets:
                self.fig.delaxes(c)

    def labels(self, x_label: str, y_label: str):
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        return self

    def region(self, x_interval, y_interval):
        self.ax.set_xlim(*x_interval)
        self.ax.set_ylim(*y_interval)
        self.ax.set_aspect('equal', adjustable='box')
        return self

    def colorbar(self, collection, label: str):
        # Create an axes for the color bar
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Add a color bar
        cbar = self.fig.colorbar(collection, cax=cax)
        cbar.set_label(label)
        return self

    def slider(self, data_length: int, update_callback):
        """
        :param update_callback: function that returns void
        """
        plt.subplots_adjust(bottom=0.2)  # reserve space for the slider
        self.ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(self.ax_slider, 'Index', 0, data_length - 1, valinit=0, valstep=1)
        self.slider.on_changed(update_callback)
        return self

    def label(self, x_label: str, y_label: str):
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        return self


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


def ListColor01(color_map_name: str, length: int):
    return plt.get_cmap(color_map_name)(np.linspace(0, 1, length))
