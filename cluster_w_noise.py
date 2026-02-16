import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN


def parse_meta_from_name(stem):
    # Parse vf / dvf / cluster count encoded in folder name
    tokens = stem.split("_")
    vf = int(tokens[1])
    dvf = int(tokens[3])
    num = int(tokens[5])
    return vf, dvf, num


def load_train_mask_from_mat(mat_path):
    # Load binary particle mask from .mat file
    ms = sio.loadmat(mat_path)["TrainData"][0][0][2]
    return ms


def mask_to_xy(mask2d):
    # Convert binary mask to particle coordinate list
    xy = np.argwhere(mask2d == 1).astype(float) #(N,2)
    if xy.size == 0:
        return xy, np.array([]), np.array([])
    return xy, xy[:, 0], xy[:, 1]


def periodic_precomputed_distance(xy, L=400.0):
    # Pairwise distance matrix with periodic boundary correction
    if xy.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)

    total = None
    for d in range(xy.shape[1]):
        pd = pdist(xy[:, d].reshape(-1, 1))
        pd[pd > (L * 0.5)] -= L
        total = pd**2 if total is None else total + pd**2

    return squareform(np.sqrt(total)) #(N,N)


def dbscan_search(square, vf, dvf, num, tol=0.1):
    # Grid search over DBSCAN parameters to match target cluster count
    bestscore=float('inf')
    best = None
    for n in range(100, 140): #min_samples: min# of points required in radius of a point to be core point
        for m in np.arange(8.0, 11.0, 0.1): # eps: neighborhood radius
            db = DBSCAN(eps=float(m), min_samples=int(n),
                        metric="precomputed").fit(square) # use distant matrix
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = int(np.sum(labels == -1))

            guess_avf = (vf / 100.0) - (n_noise_ / 160000.0)
            true_avf = (vf - dvf) / 100.0
            score_avf = guess_avf / true_avf if true_avf != 0 else np.inf #ideally 1

            ok = (n_clusters_ == num) and (abs(score_avf - 1.0) < tol) # two srandard to end searching
            if ok:
                return db, labels, n_clusters_, n_noise_, score_avf, int(n), float(m), 1

            if (n_clusters_ == num) and abs(score_avf - 1.0) < bestscore:
                best = (db, labels, n_clusters_, n_noise_,
                        score_avf, int(n), float(m), 0)
                bestscore=abs(score_avf - 1.0)

    return best


def list_samples(sample_data_dir):
    # Discover valid sample folders containing matching .mat files
    sample_data_dir = os.path.abspath(sample_data_dir)
    subs = sorted(
        d for d in os.listdir(sample_data_dir)
        if os.path.isdir(os.path.join(sample_data_dir, d))
    )
    samples = []
    for sub in subs:
        mat_path = os.path.join(sample_data_dir, sub, sub + ".mat") #/data/VF.../VF....mat
        if os.path.exists(mat_path) and sub.startswith("VF_"):
            samples.append((sub, mat_path)) #[("VF...", "/data/VF.../VF....mat"),...
    return samples


def Cluster(sample_data_dir, tol=0.1, plot_n=3):
    samples = list_samples(sample_data_dir)

    vf_list, dvf_list, num_list = [], [], []
    correct_list, min_sizes_list, max_distances_list = [], [], []
    plot_items = []

    for stem, mat_path in samples:
        vf, dvf, num = parse_meta_from_name(stem)
        ms = load_train_mask_from_mat(mat_path)

        xy, x, y = mask_to_xy(ms)
        square = periodic_precomputed_distance(xy, L=400.0)

        db, labels, n_clusters_, n_noise_, score_avf, n, m, ok = dbscan_search(
            square, vf=vf, dvf=dvf, num=num, tol=tol
        )

        vf_list.append(vf)
        dvf_list.append(dvf)
        num_list.append(num)
        correct_list.append(ok)
        min_sizes_list.append(n)
        max_distances_list.append(m)

        # Save clustering labels per sample
        out_dir = os.path.join(os.path.abspath(sample_data_dir), stem)
        np.savetxt(os.path.join(out_dir, "cluster_label.txt"),
                   labels, fmt="%d")

        print("Estimated number of clusters: %d" % n_clusters_)
        print("min_size:", n, "max_distance:", m)
        print(vf, dvf, num, end=" ")
        print("score_avf", score_avf, end=" ")
        print()

        if len(plot_items) < plot_n and labels is not None and x.size == labels.size:
            plot_items.append((stem, x, y, labels))

    vf_arr = np.array(vf_list, dtype=int)
    dvf_arr = np.array(dvf_list, dtype=int)
    num_arr = np.array(num_list, dtype=int)
    correct_arr = np.array(correct_list, dtype=int)
    min_sizes_arr = np.array(min_sizes_list, dtype=int)
    max_distances_arr = np.array(max_distances_list, dtype=float)

    # Save summary parameters across all samples
    out = np.column_stack(
        (vf_arr, dvf_arr, num_arr,
         correct_arr, min_sizes_arr, max_distances_arr)
    )
    np.savetxt(
        os.path.join(os.path.abspath(sample_data_dir), "parameter.txt"),
        out,
        fmt="%d %d %d %d %d %.5f",
        delimiter=",",
        header="vf,dvf,num,correct,min_size,max_distance",
    )

    print("good clustering=", correct_arr,
          "total samples=", len(num_arr), end=" ")
    print()

    # Visualize selected samples after full processing
    if plot_items:
        k = len(plot_items)
        fig, axes = plt.subplots(1, k, figsize=(5 * k, 5))
        if k == 1:
            axes = [axes]
        for ax, (stem, x, y, labels) in zip(axes, plot_items):
            ax.scatter(x, y, c=labels.astype(float), s=1)
            ax.set_title(stem)
            ax.set_aspect("equal")
            ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    return (vf_arr, dvf_arr, num_arr,
            min_sizes_arr, max_distances_arr, correct_arr)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "sample_data")
    Cluster(DATA_DIR, tol=0.1, plot_n=2)
