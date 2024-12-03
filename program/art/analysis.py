import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from scipy.interpolate import CubicSpline

import src.utils as ut
from src.myio import DataSet
from src.render import StateRenderer
import src.art as art


class DistSlice:
    def __init__(self, dist_array: np.ndarray):
        self.data = dist_array

    def __str__(self):
        plt.plot(self.data)
        plt.show()
        return '<DistSlice object>'


class Distribution:
    def __init__(self, mat: np.ndarray):
        self.data = mat

    def __str__(self):
        lst = self.data.T.tolist()
        art.plotListOfArray(ut.sample_equal_stride(lst, 20))
        return '<Distribution object>'

    def showAsImage(self):
        plt.imshow(self.data)
        plt.show()

    def slice(self, index: int):
        return DistSlice(self.data[:, index])

    def final(self):
        return self.slice(self.data.shape[1] - 1)


class CurveManager:
    def __init__(self, abstract: pd.DataFrame, xs: list[np.ndarray], data: list[np.ndarray]):
        self.abstract = abstract
        self.original_xs = xs
        self.original_data = data
        self._xs = None
        self._curves = None
        self.average_flags = None
        self.std_curves = None
        self.x_label = None
        self.y_label = None

    def set_average_flags(self, flags: list[str]):
        self.average_flags = flags
        return self

    def set_std_curve(self, curves: list[np.ndarray]):
        self.std_curves = curves
        return self

    @classmethod
    def derived(cls, abstract: pd.DataFrame, xs: np.ndarray, curves: np.ndarray):
        obj = cls(abstract, None, None)
        obj._xs = xs
        obj._curves = curves
        return obj

    def set_labels(self, x_label: str, y_label: str):
        self.x_label = x_label
        self.y_label = y_label
        return self

    @property
    def xs(self):
        dx = 1e-4
        if self._xs is not None:
            return self._xs
        min_x = min([x.min() for x in self.original_xs])
        max_x = max([x.max() for x in self.original_xs])
        return np.arange(min_x, max_x, dx)

    @property
    def curves(self):
        if self._curves is not None:
            return self._curves
        interpolated_data = []
        for x, y in zip(self.original_xs, self.original_data):
            cs = CubicSpline(x, y, extrapolate=False)
            new_y = cs(self.xs)
            interpolated_data.append(new_y)
        return np.array(interpolated_data)

    def _averageBy(self, merge_list: list[tuple[int]]) -> (np.ndarray, np.ndarray):
        """
        return: (average, standard error)
        Note: standard_error = standard_deviation_of_(n-1) / sqrt(n)
        """
        new_curves = []
        std_errs = []
        for merge in merge_list:
            curve_slice = np.array([self.curves[i] for i in merge])
            ave_curve = np.nanmean(curve_slice, axis=0)
            std_curve = ut.standard_error(curve_slice, axis=0)
            new_curves.append(ave_curve)
            std_errs.append(std_curve)
        return np.array(new_curves), np.array(std_errs)

    def average(self, props: str) -> 'CurveManager':
        # 'props' should be interpreted as a list[str]
        props = list(map(lambda x: x.strip(' '), props[1:-1].split(',')))
        ave, std = self._averageBy(ut.indicesOfTheSame(self.abstract, props))
        return CurveManager.derived(
            ut.groupAndMergeRows(self.abstract, props), self.xs, ave
        ).set_average_flags(props).set_std_curve(std)

    def __str__(self):
        cmap_s = 'cool'
        cmap = plt.get_cmap(cmap_s)
        colors = cmap(np.linspace(0, 1, len(self.curves)))
        for i in range(len(self.curves)):
            curve = self.curves[i]
            color = colors[i]
            if self.std_curves is not None:
                std = self.std_curves[i]
                plt.fill_between(self.xs, curve - std, curve + std, color=color, alpha=0.2)
            plt.plot(self.xs, curve, color=color)
        if self.average_flags is not None and len(self.average_flags) == 1:
            flag = self.average_flags[0]
            lst = np.asarray(self.abstract[flag])
            min_v, max_v = np.min(lst), np.max(lst)
            norm = Normalize(vmin=min_v, vmax=max_v)
            sm = ScalarMappable(cmap=cmap_s, norm=norm)
            cbar = plt.colorbar(sm)
            cbar.set_label(flag)
        if self.x_label is not None:
            plt.xlabel(self.x_label)
        if self.y_label is not None:
            plt.ylabel(self.y_label)
        plt.show()
        return f'<CurveManager of {len(self.curves)} curve(s).>'

    def export(self):
        # export curves to a csv file
        pd.DataFrame(self.curves).to_csv('output.csv', header=None, index=None)
        return self


class RenderPipe:
    def __init__(self, *funcs):
        self.sequence = list(map(ut.reverseClassMethod, funcs))

    def eval(self, obj):
        return ut.applyPipeline(obj, self.sequence)


class InteractiveViewer:
    def __init__(self, dataset: DataSet, pipe: RenderPipe, *args):
        self.metadata = dataset.metadata
        self.data = list(map(lambda x: StateRenderer(x, *args), dataset.data))
        self.index = 0
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.handle = (self.fig, self.ax)
        plt.subplots_adjust(bottom=0.2)  # reserve space for the slider
        self.ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(self.ax_slider, 'Index', 0, len(self.data) - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.pipe = pipe

    def clear(self):
        self.ax.clear()
        for c in self.fig.get_axes():
            if c is not self.ax and c is not self.ax_slider:
                self.fig.delaxes(c)

    def on_key_press(self, event):
        if event.key == 'left':
            self.index = max(0, self.index - 1)
        elif event.key == 'right':
            self.index = min(len(self.data) - 1, self.index + 1)
        else:
            return
        self.redraw()
        self.slider.set_val(self.index)

    def update(self, val):
        self.index = int(self.slider.val)
        self.redraw()

    def redraw(self):
        self.clear()
        self.data[self.index].drawBoundary(self.handle)
        self.data[self.index].drawParticles(self.handle, self.pipe.eval(self.data[self.index]))
        plt.draw()

    def show(self):
        self.redraw()
        plt.show()


if __name__ == '__main__':
    ds = DataSet.loadFrom('../data.h5')
    InteractiveViewer(ds, RenderPipe(StateRenderer.angle), True).show()
    # art.plotListOfArray(ds.descentCurves)
