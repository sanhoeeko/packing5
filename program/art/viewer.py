import matplotlib.collections as collections
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt

from analysis.analysis import OrderParameterFunc
from analysis.database import PickledSimulation
from art.art import Figure
from . import art


class RenderSetup:
    def __init__(self, order_parameter_name: str = None, weighted=False, style: str = 'default', real_size=False):
        self.style = style
        self.real_size = real_size

        def func(x):
            arr = OrderParameterFunc([order_parameter_name], weighted, False)(x)
            return arr[order_parameter_name]

        self.func = func


def selectCmapAndNorm(style: str):
    """
    :return: cmap, norm
    """
    try:
        return {
            'default': ('viridis', None),
            'angle': ('hsv', mcolors.Normalize(vmin=0, vmax=np.pi)),
            'voronoi': (mcolors.ListedColormap(art.my_colors), mcolors.Normalize(vmin=0, vmax=len(art.my_colors)))
        }[style]
    except KeyError:
        raise TypeError("Unknown visualization style.")


class RenderState:
    def __init__(self, handle: Figure):
        self.handle = handle

    def drawBoundary(self, A: float, B: float):
        ellipse = patches.Ellipse((0, 0), width=2 * A, height=2 * B, fill=False)
        self.handle.ax.add_artist(ellipse)
        self.A, self.B = A, B
        return self

    def drawParticles(self, setup: RenderSetup, xyt: np.ndarray, metadata: dict):
        """
        :param xyt: (N, 3) configuration
        :param metadata: dict converted from 1 x 1 struct array
        """
        assert hasattr(self, 'A'), "You may forget to set the boundary."
        cmap, norm = selectCmapAndNorm(setup.style)

        # Create a list to hold the patches
        ellipses = []

        # For each point in the data, create a custom patch (ellipse) and add it to the list
        if setup.real_size:
            a, b = np.sqrt(2), np.sqrt(2) / metadata['gamma']
            alpha = 0.7
        else:
            a, b = 1, 1 / metadata['gamma']
            alpha = 1.0
        for xi, yi, ti in xyt:
            # ellipse = patches.Ellipse((xi, yi), width=self.a, height=self.b, angle=ti)
            ellipse = art.Capsule((xi, yi), width=a, height=b, angle=180 / np.pi * ti)
            ellipses.append(ellipse)

        # Calculate color_data, which determines the color to display on particles
        abg = (metadata['A'], metadata['B'], metadata['gamma'])
        color_data = setup.func((abg, xyt)).data

        # Create a collection with the ellipses and add it to the axes
        col = collections.PatchCollection(ellipses, array=color_data, norm=norm, cmap=cmap, alpha=alpha)
        self.handle.ax.add_collection(col)

        # Set the limits of the plot
        self.handle.ax.set_xlim(-self.A - 1, self.A + 1)
        self.handle.ax.set_ylim(-self.B - 1, self.B + 1)
        self.handle.ax.set_aspect('equal')

        # Add a text [at the top left side] to show information of the state
        self.handle.ax.text(
            -self.A, self.B * 1.1,
            (
                f"n={metadata['n']}, d={'{:.3f}'.format(metadata['d'])}, "
                f"φ={'{:.3f}'.format(metadata['phi'])}, "
                f"A={'{:.2f}'.format(metadata['A'])}, B={'{:.2f}'.format(metadata['B'])}, "
                f"\n"
                f"E={'{:.3f}'.format(metadata['energy'])}, g={'{:.3f}'.format(metadata['gradient_amp'])}"
            )
        )
        self.handle.colorbar(col, 'θ')
        return self


class InteractiveViewer:
    def __init__(self, data: PickledSimulation, setup: RenderSetup):
        self.simu = data
        self.index = 0
        self.handle = Figure().slider(len(self.simu), self.update)
        self.handle.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.renderer = RenderState(self.handle)
        self.setup = setup

    def on_key_press(self, event):
        if event.key == 'left':
            self.index = max(0, self.index - 1)
        elif event.key == 'right':
            self.index = min(len(self.simu) - 1, self.index + 1)
        else:
            return
        self.redraw()
        self.handle.slider.set_val(self.index)

    def update(self, val):
        self.index = int(self.handle.slider.val)
        self.redraw()

    def redraw(self):
        self.handle.clear()
        dic = self.simu[self.index]
        self.renderer.drawBoundary(dic['metadata']['A'], dic['metadata']['B'])
        self.renderer.drawParticles(self.setup, dic['xyt'], dic['metadata'])
        plt.draw()

    def show(self):
        self.redraw()
        plt.show()
