import matplotlib.collections as collections
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
import scipy.spatial as sp
from matplotlib import pyplot as plt

import analysis.utils as ut
from analysis.analysis import OrderParameterFunc
from analysis.database import PickledSimulation
from analysis.voronoi import Voronoi
from art.art import Figure
from . import art
from .delaunay_art import showTypedDelaunay

style_dict = {
    'angle': ['Angle', 'DirectorAngle', 'PureRotationAngle'],
    'voronoi': ['z_number'],
    'pm1': ['S_global', 'CrystalNematicAngle'],
    'defect_pm2': ['winding2'],
}


def get_style(order_parameter_name: str) -> str:
    for k, v_list in style_dict.items():
        if order_parameter_name in v_list:
            return k
    if order_parameter_name.startswith('director-'):
        return 'angle'
    return 'default'


class RenderSetup:
    def __init__(self, order_parameter_name: str = None, real_size=True):
        self.name = order_parameter_name
        self.style = get_style(order_parameter_name)
        self.real_size = real_size

        if order_parameter_name is None:
            self.style = 'single color'
            self.func = None
        else:
            def func(x):
                arr = OrderParameterFunc([order_parameter_name], 'None')(x)
                return arr[order_parameter_name]

            self.func = func


def selectCmapAndNorm(style: str):
    """
    :return: cmap, norm
    """
    single_color = (0, 0.78, 0.625)
    try:
        return {
            'single color': (mcolors.LinearSegmentedColormap.from_list("single_color_cmap",
                                                                       [(0, single_color), (1, single_color)]),
                             mcolors.Normalize(vmin=0, vmax=1)),
            'default': ('jet', None),
            'angle': ('hsv', mcolors.Normalize(vmin=0, vmax=np.pi)),
            'pm1': ('bwr', mcolors.Normalize(vmin=-1, vmax=1)),
            'defect_pm2': ('Spectral', mcolors.Normalize(vmin=-3, vmax=3)),
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

    def drawParticles(self, xyt: np.ndarray, metadata: dict, setup: RenderSetup = None, with_label=True):
        """
        :param xyt: (N, 3) configuration
        :param metadata: dict converted from 1 x 1 structured array, must include "gamma"
        """
        external_option = None
        if not hasattr(self, 'A'):
            self.A = np.max(np.abs(xyt[:, 0]))
            self.B = np.max(np.abs(xyt[:, 1]))
        if type(setup) != RenderSetup:
            external_option = str(setup)
            setup = RenderSetup()
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
        if setup.func is None:
            color_data = np.ones((xyt.shape[0],))
        else:
            abg = (metadata['A'], metadata['B'], metadata['gamma'])
            color_data = setup.func((abg, xyt))

        # Create a collection with the ellipses and add it to the axes
        col = collections.PatchCollection(ellipses, array=color_data, norm=norm, cmap=cmap, alpha=alpha)
        self.handle.ax.add_collection(col)

        # Set the limits of the plot
        self.handle.ax.set_xlim(-self.A - 1, self.A + 1)
        self.handle.ax.set_ylim(-self.B - 1, self.B + 1)
        self.handle.ax.set_aspect('equal')

        # Add a text [at the top left side] to show information of the state
        if with_label:
            self.handle.ax.text(
                -self.A, self.B * 1.1,
                (
                    f"n={metadata['n']}, d={'{:.3f}'.format(metadata['d'])}, "
                    f"φ={'{:.3f}'.format(metadata['phi'])}, "
                    f"A={'{:.2f}'.format(metadata['A'])}, B={'{:.2f}'.format(metadata['B'])}, "
                    f"\n"
                    f"E={'{:.3f}'.format(metadata['energy'])}, g={'{:.3f}'.format(metadata['mean_gradient_amp'])}"
                )
            )
        if external_option is None:
            if setup.style == 'voronoi':
                cmap, norm = selectCmapAndNorm(setup.style)
                art.add_energy_level_colorbar(self.handle.ax, cmap, np.arange(11), 'neighbors', digits=0)
            else:
                self.handle.colorbar(col, setup.name)
        if external_option == 'typed delaunay':
            delaunay = Voronoi(metadata['gamma'], metadata['A'], metadata['B'], xyt).delaunay()
            showTypedDelaunay(self.handle, delaunay, xyt)
        elif external_option == 'full delaunay':
            voro = Voronoi(metadata['gamma'], metadata['A'], metadata['B'], xyt)
            delaunay = sp.Delaunay(voro.disk_map)
            xy = voro.disk_map
            self.handle.ax.triplot(xy[:, 0], xy[:, 1], delaunay.simplices.copy())
        return self

    def drawMarkers(self, xyt: np.ndarray, metadata: dict):
        marker_list = ['+', '1', '', r'$\perp$', 's']
        xyt_c = ut.CArray(xyt)
        winding_number_2 = Voronoi(metadata['gamma'], metadata['A'], metadata['B'], xyt).delaunay().winding2(xyt_c)
        for i in range(metadata['N']):
            if winding_number_2[i] != 0:
                x, y = xyt[i, 0:2]
                self.handle.ax.scatter(x, y, marker=marker_list[winding_number_2[i] + 2], color='black')
        return self

    def drawBonds(self, xyt: np.ndarray, edges: np.ndarray, color='black'):
        """
        edges: (n, 2) integer array (treated as list of pairs).
        """
        assert edges.shape[1] == 2
        for s, e in edges:
            x1, y1 = xyt[s, :2]
            x2, y2 = xyt[e, :2]
            self.handle.ax.plot([x1, x2], [y1, y2], color=color)


class InteractiveViewer:
    def __init__(self, data: PickledSimulation, setup: RenderSetup):
        self.simu = data
        self.index = 0
        self.handle = Figure().slider(len(self.simu), self.update)
        self.handle.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.renderer = RenderState(self.handle)
        self.setup = setup
        self.marker_setup = []

    def setMarkerSetup(self, setup: str):
        """
        :param setup: string of tags separated by blanks. Example: "lc-defect new-bonds"
        """
        self.marker_setup = setup.split(' ')
        return self

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
        self.renderer.drawParticles(dic['xyt'], dic['metadata'], self.setup)
        if 'lc-defect' in self.marker_setup:
            self.renderer.drawMarkers(dic['xyt'], dic['metadata'])
        if 'new-bonds' in self.marker_setup:
            if self.index > 0:
                # testDefectEvents(dic, self.simu[self.index - 1])
                pairs = getDefectTracks(dic, self.simu[self.index - 1])
                self.renderer.drawBonds(dic['xyt'], pairs)
        if 'old-bonds' in self.marker_setup:
            if self.index > 0:
                pairs = getDefectTracksNegative(dic, self.simu[self.index - 1])
                self.renderer.drawBonds(dic['xyt'], pairs, color='red')
        plt.draw()

    def show(self):
        self.redraw()
        plt.show()


def getDefectTracks(current_state_dic: dict, previous_state_dic: dict) -> np.ndarray[np.int32]:
    return Voronoi.fromStateDict(current_state_dic).delaunay().difference(
        Voronoi.fromStateDict(previous_state_dic).delaunay()
    ).toPairs()


def getDefectTracksNegative(current_state_dic: dict, previous_state_dic: dict) -> np.ndarray[np.int32]:
    return Voronoi.fromStateDict(previous_state_dic).delaunay().difference(
        Voronoi.fromStateDict(current_state_dic).delaunay()
    ).toPairs()


def testDefectEvents(current_state_dic: dict, previous_state_dic: dict):
    d1 = Voronoi.fromStateDict(current_state_dic).delaunay()
    d0 = Voronoi.fromStateDict(previous_state_dic).delaunay()
    xyt_1 = ut.CArray(current_state_dic['xyt'])
    xyt_0 = ut.CArray(previous_state_dic['xyt'])
    events = d1.events_compared_with(d0, xyt_1, xyt_0)
    print(events)
