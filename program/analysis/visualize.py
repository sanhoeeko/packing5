from art.curves import plotMeanCurvesWithCI
from .h5tools import read_hdf5_groups_to_dicts


class AnalysisData:
    """
    This class parses `analysis.h5`
    """
    def __init__(self, file_name: str):
        self.file_name = file_name
        dic, self.order_parameters = read_hdf5_groups_to_dicts(file_name)
        self.x_axis_name = str(dic['x_axis_name'], encoding='utf-8')
        self.x_axis = dic['x_axis']

        # some metadata
        super().__init__(dic)

    def plot(self, order_parameter_name: str):
        mean = self.order_parameters[order_parameter_name]['mean']
        ci = self.order_parameters[order_parameter_name]['ci']
        plotMeanCurvesWithCI(self.x_axis, mean, ci, self.x_axis_name, order_parameter_name)
