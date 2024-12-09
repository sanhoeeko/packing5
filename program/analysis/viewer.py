from matplotlib.widgets import Slider
from src.myio import DataSet
from src.render import StateRenderer


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
