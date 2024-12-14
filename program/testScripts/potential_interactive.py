from testScripts.kernel_for_test import TestPotential, setWorkingDirectory

setWorkingDirectory()

from math import pi, sin, cos

import matplotlib.pyplot as plt
import numpy as np

from art.art import Capsule
from simulation.potential import Potential, PowerFunc


class XytPair:
    def __init__(self, lst):
        assert len(lst) == 6
        self.f1 = lst[0:2]
        self.m1 = lst[2]
        self.f2 = lst[3:5]
        self.m2 = lst[5]

    @property
    def first(self):
        return self.f1, self.m1

    @property
    def second(self):
        return self.f2, self.m2


class MyCapsule:
    def __init__(self, n, d, x, y, theta, color='violet'):
        self.n, self.d = n, d
        self.a, self.b = 1, 1 / (1 + (n - 1) * d / 2)
        self.x, self.y, self.theta = x, y, theta
        self.color = color

    @property
    def center(self):
        return np.array([self.x, self.y])

    def show(self):
        return Capsule((self.x, self.y), width=np.sqrt(2) * self.a, height=np.sqrt(2) * self.b,
                       angle=180 * self.theta / pi,
                       color=self.color, alpha=0.7)

    def notice(self, x, y):
        if (x - self.x) ** 2 + (y - self.y) ** 2 < self.b ** 2:
            # print(f"Noticed. Center at: ({self.x}, {self.y})")
            return True
        return False

    def moveTo(self, x, y):
        self.x, self.y = x, y

    def rotate(self, angle):
        self.theta += angle


class Shape:
    def __init__(self, potential):
        self.func = potential.interpolateGradient

    def reference(self):
        return GradientReference()


class GradientReference:
    def __init__(self):
        self.func = potential.preciseGradient


def rotForce(f: np.ndarray, t) -> np.ndarray:
    U = np.array([[cos(t), -sin(t)], [sin(t), cos(t)]])
    return (U @ f.reshape((2, 1))).reshape(-1)


class ForceInterface:
    def __init__(self, e1: MyCapsule, e2: MyCapsule):
        assert e1.n == e2.n, e1.d == e2.d
        self.shape = Shape(potential)
        self.ref = self.shape.reference()

    def calForce(self, dx, dy, t1, t2):
        xyt_pair = XytPair(self.shape.func(dx, dy, t1, t2))
        return xyt_pair

    def calForceReference(self, dx, dy, t1, t2):
        xyt_pair = XytPair(self.ref.func(dx, dy, t1, t2))
        return xyt_pair

    def calPoint(self, theta, center, F, moment):
        u = np.array([cos(theta), sin(theta)])
        r = moment / (np.cross(u, F))
        force_act_on_point = center + r * u
        return force_act_on_point

    def force(self, e1: MyCapsule, e2: MyCapsule) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

        # a, b = e1.a, e1.b  # suppose that two capsules are identity
        dx = e1.x - e2.x
        dy = e1.y - e2.y

        if mode == 'test':
            xyt_pair = self.calForce(dx, dy, e1.theta, e2.theta)
        elif mode == 'ref':
            xyt_pair = self.calForceReference(dx, dy, e1.theta, e2.theta)

        F1, M1 = xyt_pair.first
        P1 = self.calPoint(e1.theta, e1.center, F1, M1)
        F2, M2 = xyt_pair.second
        P2 = self.calPoint(e2.theta, e2.center, F2, M2)
        return F1, P1, F2, P2

    def rePlot(self):
        ax.cla()
        for obj in objs:
            ax.add_artist(obj.show())
        # plot force
        force1, point1, force2, point2 = self.force(objs[0], objs[1])
        if np.linalg.norm(force1) > 0:
            drawPlacedVector(force1, point1)
        if np.linalg.norm(force2) > 0:
            drawPlacedVector(force2, point2)
        # set ax range
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

        plt.draw()


def drawPlacedVector(vector: np.ndarray, pos: np.ndarray):
    # print(f"Draw force: ({pos[0]}, {pos[1]}), ({vector[0]}, {vector[1]})")
    ax.arrow(pos[0], pos[1], vector[0] * 0.1, vector[1] * 0.1, head_width=0.1, head_length=0.1)


class Handler:
    def __init__(self):
        self.current_obj = None

    def notice(self, x, y):
        if self.current_obj is None:
            for obj in objs:
                if obj.notice(x, y):
                    self.current_obj = obj
                    break
        else:
            self.release()

    def release(self):
        # print(f"Placed. Center at: ({self.current_obj.x}, {self.current_obj.y})")
        self.current_obj = None

    def moveTo(self, x, y):
        if self.current_obj is not None:
            self.current_obj.moveTo(x, y)
            fi.rePlot()

    def rotatePlus(self):
        if self.current_obj is not None:
            self.current_obj.rotate(0.1)
            fi.rePlot()

    def rotateMinus(self):
        if self.current_obj is not None:
            self.current_obj.rotate(-0.1)
            fi.rePlot()


handler = Handler()


def on_press(event):
    if event.button == 1:
        handler.notice(event.xdata, event.ydata)


def on_release(event):
    pass


def on_move(event):
    handler.moveTo(event.xdata, event.ydata)


def on_scroll(event):
    if event.button == 'up':
        handler.rotatePlus()
    elif event.button == 'down':
        handler.rotateMinus()


mode = 'ref'
n, d = 2, 1
potential = TestPotential(Potential(n, d, PowerFunc(2.5)).cal_potential(threads=4))

if __name__ == '__main__':
    objs = []
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    print("Click and release, then drag capsules. Scroll to spin capsules.")
    objs.append(MyCapsule(2, 1, 1, 1, 1, 'violet'))  # e1
    objs.append(MyCapsule(2, 1, 0, 0, 0, 'springgreen'))  # e2
    fi = ForceInterface(*objs)
    fi.rePlot()
    plt.show()
