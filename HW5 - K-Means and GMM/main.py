import numpy as np
import matplotlib.pyplot as plt

import kmeans
import gmm
import pca


def generate_dataset(sigma):
    p_a = np.random.multivariate_normal([-1, -1], sigma * np.asarray([[2, 0.5], [0.5, 1]]), size=100)
    p_b = np.random.multivariate_normal([1, -1], sigma * np.asarray([[1, -0.5], [-0.5, 2]]), size=100)
    p_c = np.random.multivariate_normal([0, 1], sigma * np.asarray([[1, 0], [0, 2]]), size=100)
    return np.concatenate((p_a, p_b, p_c))


def one():
    # Q 1.2
    # Generating dataset
    sigma = [0.5, 1, 2, 4, 8]
    datasets = []
    for s in sigma:
        datasets.append(generate_dataset(s))

    km_acc = []
    km_obj = []
    gmm_acc = []
    gmm_nll = []

    for idx, dataset in enumerate(datasets):
        print(f"Sigma: {sigma[idx]}")
        km = kmeans.KMeans(3, dataset)
        km_acc.append(km.accuracy())
        km_obj.append(km.objective())
        gm = gmm.GMM(3, dataset)
        gmm_acc.append(gm.accuracy())
        gmm_nll.append(gm.log_likelihood())

    plt.title("Sigma vs Accuracy")
    plt.plot(sigma, km_acc)
    plt.plot(sigma, gmm_acc)
    plt.xlabel("Sigma")
    plt.ylabel("Accuracy")
    plt.legend(["K-Means", "GMM"])
    plt.grid()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.title.set_text("K-Means Objective vs Sigma")
    ax1.plot(sigma, km_obj)
    ax1.set_xlabel("Sigma")
    ax1.set_ylabel("Objective")
    ax1.grid()

    ax2.title.set_text("GMM Negative Log Likelihood vs Sigma")
    ax2.plot(sigma, gmm_nll)
    ax2.set_xlabel("Sigma")
    ax2.set_ylabel("Negative Log Likelihood")
    ax2.grid()
    plt.show()


def two():
    data2d = np.loadtxt("data/data2d.csv", delimiter=',')
    data1000d = np.loadtxt("data/data1000d.csv", delimiter=',')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    bv, bz, bx = pca.buggy_pca(data2d, 1)
    pca.plot_pca_2d(data2d, bx, ax1)
    ax1.title.set_text("Buggy PCA")
    print(f"Buggy PCA: {pca.reconstruction_error(data2d, bx)}")

    dv, dz, dx = pca.demeaned_pca(data2d, 1)
    pca.plot_pca_2d(data2d, dx, ax2)
    ax2.title.set_text("Demeaned PCA")
    print(f"Demeaned PCA: {pca.reconstruction_error(data2d, dx)}")

    nv, nz, nx = pca.normalized_pca(data2d, 1)
    pca.plot_pca_2d(data2d, nx, ax3)
    ax3.title.set_text("Normalized PCA")
    print(f"Normalized PCA: {pca.reconstruction_error(data2d, nx)}")

    drv, drz, drx = pca.dro(data2d, 1)
    xzv = np.matmul(drx, drz.T) + drv
    pca.plot_pca_2d(data2d, xzv, ax4)
    ax4.title.set_text("DRO")
    print(f"DRO: {pca.reconstruction_error(data2d, xzv)}")

    plt.show()

    buggy_err = []
    demeaned_err = []
    normalized_err = []
    dro_err = []
    D = [i for i in range(0, 1050, 50)]
    for d in D:
        print(d)
        bv, bz, bx = pca.buggy_pca(data1000d, d)
        buggy_err.append(pca.reconstruction_error(data1000d, bx))

        dv, dz, dx = pca.demeaned_pca(data1000d, d)
        demeaned_err.append(pca.reconstruction_error(data1000d, dx))

        nv, nz, nx = pca.normalized_pca(data1000d, d)
        normalized_err.append(pca.reconstruction_error(data1000d, nx))

        drv, drz, drx = pca.dro(data2d, 1)
        xzv = np.matmul(drx, drz.T) + drv
        dro_err.append(pca.reconstruction_error(data2d, xzv))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(D, buggy_err)
    ax1.set_xlabel("d")
    ax1.set_ylabel("Reconstruction Error")
    ax1.title.set_text("Buggy PCA")

    ax2.plot(D, demeaned_err)
    ax2.set_xlabel("d")
    ax2.set_ylabel("Reconstruction Error")
    ax2.title.set_text("Demeaned PCA")

    ax3.plot(D, normalized_err)
    ax3.set_xlabel("d")
    ax3.set_ylabel("Reconstruction Error")
    ax3.title.set_text("Normalized PCA")

    ax4.plot(D, dro_err)
    ax4.set_xlabel("d")
    ax4.set_ylabel("Reconstruction Error")
    ax4.title.set_text("DRO")

    plt.show()



if __name__ == '__main__':
    # # Q1.2
    # one()

    # # Q2.3
    two()
