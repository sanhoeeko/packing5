import numpy as np
from matplotlib import patches, transforms
from matplotlib.path import Path
from scipy.signal import savgol_filter

from .art import Figure, ListColor01, add_energy_level_colorbar


def plotListOfArray(lst: np.ndarray, labels: tuple[str, str] = None, y_restriction: float = None):
    colors = ListColor01('jet', len(lst))
    with Figure() as f:
        for i in range(len(lst)):
            f.ax.plot(lst[i], color=colors[i], alpha=0.5)
        if labels is not None:
            f.labels(*labels)
        if y_restriction is not None:
            f.region(None, (0, y_restriction), False)


def plotMeanCurvesWithCI(x_lst: list[np.ndarray], y_mean_lst: list[np.ndarray], y_ci_lst: list[np.ndarray],
                         gammas: np.ndarray = None, gamma_label='gamma', x_label='', y_label=''):
    """
    x_lst can be None
    """
    colormap = 'jet'

    if x_lst is None:
        x_lst = [np.arange(len(y)) for y in y_mean_lst]
    assert len(x_lst) == len(y_mean_lst) == len(y_ci_lst)

    colors = ListColor01(colormap, len(y_mean_lst))
    with Figure(figsize=(10, 6)) as f:
        for i, (x, y_mean, y_ci) in enumerate(zip(x_lst, y_mean_lst, y_ci_lst)):
            f.ax.fill_between(x, y_mean - y_ci, y_mean + y_ci, color=colors[i], alpha=0.2)
            f.ax.plot(x, y_mean, color=colors[i])
        f.labels(x_label, y_label)
        if gammas is not None:
            add_energy_level_colorbar(f.ax, colormap, gammas, gamma_label)


def scatterList(xs_ys: list[tuple], x_name: str, y_name: str, y_restriction: float = None, gammas: np.ndarray = None,
                gamma_label='gamma'):
    colormap = 'jet'
    colors = ListColor01(colormap, len(xs_ys))
    with Figure() as fig:
        for i, xy in enumerate(xs_ys):
            x, y = xy
            fig.ax.scatter(x, y, s=2, color=colors[i], alpha=0.5)
        fig.labels(x_name, y_name)
        if y_restriction is not None:
            fig.region(None, (0, y_restriction), False)
        if gammas is not None:
            add_energy_level_colorbar(fig.ax, colormap, gammas, gamma_label)


def scatterCorrelations(x: np.ndarray, y: np.ndarray):
    """
    :param x: (samples, N) array
    :param y: (samples, N) array
    """
    assert x.shape == y.shape
    colors = ListColor01('jet', x.shape[0])
    with Figure() as f:
        for i in range(x.shape[0]):
            f.ax.scatter(x[i, :], y[i, :], color=colors[i], s=1, alpha=0.1)
        f.region([0, 1], [0, 1])


def normalize01(x: np.ndarray):
    x0 = np.min(x)
    x1 = np.max(x)
    return (x - x0) / (x1 - x0)


def arrow_plot(fig: Figure, t: np.ndarray, x: np.ndarray, y: np.ndarray, n_arrows=1, arrow_size=1, alpha=1,
               **plot_args):
    """
    t: curve parameter
    x, y: data
    plot_args: other parameters passed to plt.plot
    """

    def Smoother(window_size: int, order: int):
        def inner(x: np.ndarray):
            return savgol_filter(x, window_size, order)

        return inner

    smooth = Smoother(21, 3)
    x_sm = smooth(x)
    y_sm = smooth(y)
    line, = fig.ax.plot(x, y, alpha=alpha, **plot_args)
    color = line.get_color()

    # 计算导数场
    t = normalize01(t)
    dx_dt = np.gradient(x_sm, t)
    dy_dt = np.gradient(y_sm, t)

    # 生成分位点（保留边缘间距）
    t_values = np.quantile(t, np.linspace(0, 1, n_arrows + 2))[1:-1]

    # 创建三角形模版（原始指向右方）
    triangle_verts = np.array([
        [0.5, 0.0],  # 顶点（向右延伸）
        [-0.5, -0.3],  # 左下方（调整宽度保持黄金比例）
        [-0.5, 0.3],  # 左上方
        [0.5, 0.0]  # 闭合路径
    ]) * 1.618  # 缩放系数补偿
    triangle_codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    triangle_path = Path(triangle_verts, triangle_codes)

    for t_val in t_values:
        # 插值获取坐标和方向
        xi = np.interp(t_val, t, x)
        yi = np.interp(t_val, t, y)
        dx = np.interp(t_val, t, dx_dt)
        dy = np.interp(t_val, t, dy_dt)

        # 计算旋转角度
        angle = np.degrees(np.arctan2(dy, dx))

        # 创建复合变换
        transform = transforms.Affine2D().scale(
            arrow_size,
            arrow_size * 0.618
        ).rotate_deg(
            angle
        ).translate(
            xi, yi  # 此时平移的是重心位置
        ) + fig.ax.transData

        # 添加三角形补丁
        patch = patches.PathPatch(
            triangle_path,
            facecolor=color,
            edgecolor='none',  # 移除边框
            transform=transform,
            zorder=3,  # 确保箭头在曲线上方
            alpha=alpha
        )
        fig.ax.add_patch(patch)
    return fig
