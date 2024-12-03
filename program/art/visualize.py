import glob
import os
from functools import lru_cache

import matplotlib.pyplot as plt
import pandas as pd

import src.art as art
import src.utils as ut
from analysis import InteractiveViewer, RenderPipe, CurveManager, Distribution
from src.myio import DataSet
from src.render import StateRenderer
from src.state import State


# in PyCharm: File -> Settings -> Project -> Project Structure -> Add content root
# delete the old root and add "/code" as a root


class DataViewer:
    def __init__(self, datasets: list[DataSet]):
        self.datasets = list(filter(lambda x: len(x) > 0, datasets))
        self.initDensityCurveTemplates()
        self.sort()

    def name(self, Id: str) -> DataSet:
        return ut.findFirst(self.datasets, lambda x: x.id == Id)

    @property
    def _abstract(self):
        df = pd.DataFrame()
        for d in self.datasets:
            df = pd.concat([df, d.toDataFrame()], ignore_index=True)
        return df.round(5)  # 5 digits, for comparison between floats

    @property
    def abstract(self) -> pd.DataFrame:
        return self.sortedAbstract(('potential', 'gamma', 'n', 'Gamma0'))

    @lru_cache(maxsize=None)
    def sortedAbstract(self, props: tuple):
        return self._abstract.sort_values(by=list(props)).reset_index(drop=True)

    def parse(self, cmd: str):
        parser = ut.CommandQueue(self)
        tokens = cmd.split()
        for token in tokens:
            parser.push(token)
        return parser.result()

    def sort(self):
        self.datasets = ut.sortListByDataFrame(self.abstract, self.datasets)

    def print(self):
        print(self.abstract)
        return self

    def printall(self):
        print(self.abstract.to_string())
        return self

    def filter(self, key: str, value: str) -> 'DataViewer':
        ds = list(filter(
            lambda dataset: str(getattr(dataset, key)) == value,
            self.datasets))
        if len(ds) == 0:
            print('No data that matches the case!')
        return DataViewer(ds)

    def take(self, kvs: str):
        x = self
        for kv in kvs.split(','):
            x = x.filter(*kv.split('='))
        return x.print()

    def potential(self, potential_nickname: str):
        dic = {
            'hz': 'Hertzian',
            'sc': 'ScreenedCoulomb',
        }
        return self.filter('potential', dic[potential_nickname]).print()

    def render(self, Id: str, render_mode: str, *args):
        real = True if len(args) > 0 and args[0] == 'real' else False
        InteractiveViewer(self.name(Id), RenderPipe(getattr(StateRenderer, render_mode)), real).show()

    def show(self, Id: str):
        self.render(Id, 'angle')

    def curveVsDensityTemplate(self, prop: str):
        def Y(Id: str):
            y = self.name(Id).curveTemplate(prop)
            rhos = self.name(Id).rhos
            plt.plot(rhos, y)
            plt.xlabel('number density')
            plt.ylabel(prop)
            plt.show()

        return Y

    def initDensityCurveTemplates(self):
        for prop in ['energy', 'logE', 'residualForce', 'globalS', 'globalSx', 'meanS',
                     'meanDistance', 'meanZ', 'finalStepSize', 'entropyOfAngle',
                     'Phi4', 'Phi6']:
            setattr(self, prop, self.curveVsDensityTemplate(prop))

    def allTemplate(self, flag: str):
        """
        `flag` can be 'rhos' or 'phis'
        """
        flag_name = {
            'rhos': 'number density',
            'phis': 'area fraction',
        }[flag]

        def inner(prop: str):
            curves = []
            for d in self.datasets:
                curves.append((getattr(d, flag), d.curveTemplate(prop)))
            return CurveManager(self.abstract, *list(zip(*curves))).set_labels(flag_name, prop)

        return inner

    def all(self, prop: str):
        return self.allTemplate('rhos')(prop)

    def allphi(self, prop: str):
        return self.allTemplate('phis')(prop)

    def density(self, Id: str, density: float) -> State:
        density = float(density)
        return self.name(Id).stateAtDensity(density)

    def critical(self, Id: str, energy_threshold: str) -> State:
        energy_threshold = float(energy_threshold)
        return self.name(Id).critical(energy_threshold)

    def angleDist(self, Id: str) -> Distribution:
        return Distribution(self.name(Id).angleDistribution())

    def SiDist(self, Id: str) -> Distribution:
        return Distribution(self.name(Id).SiDistribution())

    def desCurve(self, Id: str):
        curves = self.name(Id).descentCurves
        curves = [cur / cur[0] - 1 if len(cur) > 1 else None for cur in curves]
        curves = list(filter(lambda x: x is not None, curves))
        art.plotListOfArray(curves)

    def distanceCurve(self, Id: str):
        d = self.name(Id)
        plt.plot(d.rhos[1:], d.distanceCurve)
        plt.xlabel('number density')
        plt.ylabel('distance between two states')
        plt.show()


def collectResultFiles(path: str):
    files = []
    for root, _, _ in os.walk(path):
        files.extend(glob.glob(os.path.join(root, '*.h5')))
    return [os.path.abspath(file) for file in files]


def loadAll(target_dir: str):
    data_files = collectResultFiles(target_dir)
    ds = ut.Map('Debug')(DataSet.loadFrom, data_files)
    ds = list(filter(lambda x: x is not None, ds))
    return DataViewer(ds)
