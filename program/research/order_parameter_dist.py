import pickle as pkl
from multiprocessing import Pool

import numpy as np

import analysis.utils as ut
from analysis.database import Database

filenames = ut.filenamesFromTxt('all-data-files-local.txt')

bins = 100


# 将内部循环封装为可并行处理的函数
def process_gamma(gamma, filenames, bins):
    dbs = [Database(file) for file in filenames]
    lst = []
    for idx in range(len(dbs[0].find(gamma=gamma)[0][0])):
        hist_tot = 0
        for i in range(len(dbs)):
            for j in range(5):
                e = dbs[i].find(gamma=gamma)[0][j]
                xyt = ut.CArray(e[idx]['xyt'])
                Sl = e.voronoi_at(idx).delaunay().S_local(xyt)
                hist, _ = np.histogram(np.array(Sl), bins=bins, range=(0, 1))
                hist_tot += hist
        hist_tot = hist_tot.astype(float) / np.sum(hist_tot) * bins
        lst.append(hist_tot)
    return np.vstack(lst)


# 主程序
if __name__ == '__main__':  # 确保在Windows/Linux下都能安全多进程
    # 创建gamma参数列表
    gamma_list = list(np.arange(1.1, 3, 0.1))

    # 创建进程池
    num_processes = len(gamma_list)
    with Pool(processes=num_processes) as pool:
        # 并行处理所有gamma值
        llst = pool.starmap(
            process_gamma,
            [(gamma, filenames, bins) for gamma in gamma_list]
        )

    with open('hists.pkl', 'wb') as f:
        pkl.dump(llst, f)

# Single process
#
# llst = []
# for gamma in np.arange(1.1, 3, 0.1):
#     lst = []
#     for idx in range(len(dbs[0].find(gamma=gamma)[0][0])):
#         hist_tot = 0
#         for i in range(len(dbs)):
#             for j in range(5):
#                 e = dbs[i].find(gamma=gamma)[0][j]
#                 xyt = ut.CArray(e[idx]['xyt'])
#                 Sl = e.voronoi_at(idx).delaunay().S_local(xyt)
#                 hist, _ = np.histogram(np.array(Sl), bins=bins, range=(0, 1))
#                 hist_tot += hist
#         hist_tot = hist_tot.astype(float) / np.sum(hist_tot) * bins
#         # plt.scatter(np.linspace(0, 1, bins, endpoint=False), hist_tot)
#         # plt.show()
#         lst.append(hist_tot)
#     llst.append(np.vstack(lst))
