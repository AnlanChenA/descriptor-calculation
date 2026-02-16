import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion


def list_sample_folders(sample_data_dir):
    sample_data_dir = os.path.abspath(sample_data_dir)
    subs = sorted(
        d for d in os.listdir(sample_data_dir)
        if os.path.isdir(os.path.join(sample_data_dir, d)) and d.startswith("VF_") and len(d) > 10
    )
    items = []
    for stem in subs:
        mat_path = os.path.join(sample_data_dir, stem, f"{stem}.mat")
        lab_path = os.path.join(sample_data_dir, stem, "cluster_label.txt")
        if os.path.exists(mat_path) and os.path.exists(lab_path):
            items.append((stem, mat_path, lab_path))
    return items


def parse_meta_from_stem(stem):
    t = stem.split("_")
    return int(t[1]), int(t[3]), int(t[5])


def load_mask(mat_path):
    return sio.loadmat(mat_path)["TrainData"][0][0][2]


def mask_to_xy(ms):
    xy = np.argwhere(ms == 1).astype(int)
    if xy.size == 0:
        return xy, np.array([], dtype=int), np.array([], dtype=int)
    return xy, xy[:, 0], xy[:, 1] #(N,2), (N,), (N,)


def make_cross_kernel():
    m = np.zeros((3, 3), dtype=int)
    m[1, :] = 1
    m[:, 1] = 1
    return m


def labels_to_mask(x, y, idxs, H=400, W=400):
    a = np.zeros((H, W), dtype=int)
    if idxs.size == 0:
        return a
    a[x[idxs], y[idxs]] = 1
    return a


def boundary_from_mask(a, kernel, iters=0):
    b = a.astype(bool)
    if iters > 0:
        b = binary_dilation(b, structure=kernel, iterations=iters)
        b = binary_erosion(b, structure=kernel, iterations=iters) # now solid inside
    bd = binary_dilation(~b, structure=kernel) & b # find boundary
    return bd.astype(int)


def compute_ifiller(ms, labels, num_clusters, kernel):
    _, x, y = mask_to_xy(ms)

    single = np.where(labels == -1)[0] # indices of all elements in labels that are equal to -1
    single_mask = labels_to_mask(x, y, single)
    bound1 = boundary_from_mask(single_mask, kernel, iters=0) ## singles: no need to really do dilation-erosion
    count1 = int(np.count_nonzero(bound1 == 1))

    count2 = 0
    bound = bound1.copy()

    for k in range(num_clusters):
        cluster = np.where(labels == k)[0] # indices of all elements in labels that are equal to k
        cluster_mask = labels_to_mask(x, y, cluster)
        bound2 = boundary_from_mask(cluster_mask, kernel, iters=5)
        count2 += int(np.count_nonzero(bound2 == 1))
        bound = (bound | bound2).astype(int)

    return count1 + count2, bound


def run_ifiller(sample_data_dir, plot_n=2):
    samples = list_sample_folders(sample_data_dir)
    kernel = make_cross_kernel()

    vf_list, dvf_list, num_list = [], [], []
    ifillers, ifillerbars = [], []
    plot_items = []

    for stem, mat_path, lab_path in samples:
        vf, dvf, num = parse_meta_from_stem(stem)
        ms = load_mask(mat_path)
        labels = np.loadtxt(lab_path).astype(int)

        Ifiller, bound = compute_ifiller(ms, labels, num, kernel)
        Ifillerbar = Ifiller / (vf / 100.0)

        vf_list.append(vf)
        dvf_list.append(dvf)
        num_list.append(num)
        ifillers.append(Ifiller)
        ifillerbars.append(Ifillerbar)

        print(vf, dvf, num, Ifiller, Ifillerbar)

        if len(plot_items) < plot_n:
            plot_items.append((stem, bound))

    out = np.column_stack((vf_list, dvf_list, num_list, ifillers, ifillerbars))
    np.savetxt(
        os.path.join(os.path.abspath(sample_data_dir), "Ifiller_new.txt"),
        out,
        fmt="%d %d %d %d %.5f",
        delimiter=",",
        header="vf,dvf,num,Ifillers,Ifillerbars",
    )

    if plot_items:
        k = len(plot_items)
        fig, axes = plt.subplots(1, k, figsize=(7 * k, 7))
        if k == 1:
            axes = [axes]
        for ax, (stem, bound) in zip(axes, plot_items):
            bound_plot = np.rot90(bound, k=-1)   
            bound_plot = np.fliplr(bound_plot)   
            ax.imshow(bound_plot)
            ax.set_title(stem)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return out

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "sample_data")
    run_ifiller(DATA_DIR, plot_n=2)
