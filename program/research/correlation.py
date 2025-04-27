import matplotlib.pyplot as plt
import numpy as np

import analysis.mymath as mm
import analysis.utils as ut
import art.art as art
import analysis.analysis as ana
from analysis.database import PickledEnsemble, Database


def Correlation(e: PickledEnsemble, truncate_x=None, truncate_h=None):
    if truncate_h is not None:
        reference_phi = ut.reference_phi(e.index_at(0)[0]['metadata']['gamma'], truncate_h)
    elif truncate_x is not None:
        reference_phi = truncate_x
    else:
        reference_phi = None

    rs = []
    corrs = []
    for i in range(e.n_density):
        if not (i % 20 == 0):
            continue
        sub_ensemble = e.index_at(i)
        if reference_phi is not None and reference_phi < sub_ensemble[0]['metadata']['phi']:
            break
        abg = (sub_ensemble[0]['metadata']['A'], sub_ensemble[0]['metadata']['B'], sub_ensemble[0]['metadata']['gamma'])
        xyts = [dic['xyt'] for dic in sub_ensemble]
        # r_lst, corr_lst = CorrelationOverEnsemble('EllipticPhi6', 'EllipticPhi6')(abg, xyts)
        r_lst, corr_lst = ana.AngularCorrelationOverEnsemble(abg, xyts)
        r = np.hstack(r_lst)
        corr = np.hstack(corr_lst)
        r, corr = mm.bin_and_smooth(r, corr, num_bins=200, apply_gaussian=True, sigma=2)
        rs.append(r)
        corrs.append(corr)

    colors = art.ListColor01('cool', len(rs))
    for i in range(len(rs)):
        plt.plot(rs[i], corrs[i], color=colors[i])
    plt.show()


if __name__ == '__main__':
    db = Database('../data-20250419-2.h5')
    idx = 19
    e = db.find(gamma=1.1 + 0.1 * idx)[0]
    Correlation(e, truncate_h=1.2)
