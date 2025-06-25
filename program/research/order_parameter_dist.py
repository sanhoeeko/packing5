import numpy as np
from matplotlib import pyplot as plt
from art.art import ListColor01
import analysis.utils as ut
from analysis.database import Database


# 将内部循环封装为可并行处理的函数
def process_gamma(order_parameter_name: str):
    start, stop = 0, 8

    def inner(gamma, filenames, bins):
        dbs = [Database(file) for file in filenames]
        lst = []
        for idx in range(len(dbs[0].find(gamma=gamma)[0][0])):
            hist_tot = 0
            for i in range(len(dbs)):
                for j in range(5):
                    e = dbs[i].find(gamma=gamma)[0][j]
                    xyt = ut.CArray(e[idx]['xyt'])
                    op = getattr(e.voronoi_at(idx).delaunay(), order_parameter_name)(xyt)
                    hist, _ = np.histogram(np.array(op), bins=bins, range=(start, stop))
                    hist_tot += hist
            hist_tot = hist_tot.astype(float) / np.sum(hist_tot) * bins
            lst.append(hist_tot)
        return np.vstack(lst)

    return inner


# filenames = ut.filenamesFromTxt('all-data-files-local.txt')
filenames = ['../data-20250419.h5', ]
bins = 100

# 主程序
if __name__ == '__main__':  # 确保在Windows/Linux下都能安全多进程
    # # 创建gamma参数列表
    # gamma_list = list(np.arange(1.1, 3, 0.5))
    #
    # # 创建进程池
    # partial_func = partial(process_gamma('FarthestSegmentDist'), filenames=filenames, bins=bins)
    # # num_processes = len(gamma_list)
    # num_processes = 4
    # with Pool(processes=num_processes) as pool:
    #     # 并行处理所有gamma值
    #     llst = pool.map(
    #         partial_func, gamma_list
    #     )
    #
    # with open('hists.pkl', 'wb') as f:
    #     pkl.dump(llst, f)

    # Single process
    order_parameter_name = 'FarthestSegmentDist'
    dbs = [Database(file) for file in filenames]
    llst = []
    start, stop = 0, 8
    for gamma in np.arange(1.1, 3, 0.5):
        lst = []
        n = len(dbs[0].find(gamma=gamma)[0][0])
        colors = ListColor01('jet', n)
        for idx in range(0, n, 10):
            if dbs[0].find(gamma=gamma)[0][0][idx]['metadata']['phi'] > 0.84:
                break
            hist_tot = 0
            for i in range(len(dbs)):
                for j in range(5):
                    e = dbs[i].find(gamma=gamma)[0][j]
                    xyt = ut.CArray(e[idx]['xyt'])
                    op = getattr(e.voronoi_at(idx).delaunay(), order_parameter_name)(xyt)
                    hist, _ = np.histogram(np.array(op), bins=bins, range=(start, stop))
                    hist_tot += hist
            hist_tot = hist_tot.astype(float) / np.sum(hist_tot) * bins
            # plt.scatter(np.linspace(start, stop, bins, endpoint=False), hist_tot, s=2, c=colors[idx])
            plt.plot(np.linspace(start, stop, bins, endpoint=False), hist_tot, color=colors[idx], alpha=0.5)
            # plt.show()
            lst.append(hist_tot)
        llst.append(np.vstack(lst))
        plt.show()
