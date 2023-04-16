import imageio
import numpy as np
from aux_func.acc_ass import assess_accuracy
from aux_func.clustering import otsu


def fuse_DI():
    LocalGCN_DI = imageio.imread('./result/SRGCAE_VerConc_DI.png').astype(np.float32)
    NLocalGCN_DI = imageio.imread('./result/SRGCAE_EdgeConc_DI.png').astype(np.float32)
    height, width = LocalGCN_DI.shape
    alpha = np.var(LocalGCN_DI.reshape(-1))
    beta = np.var(NLocalGCN_DI.reshape(-1))
    fuse_DI = (alpha * LocalGCN_DI + beta * NLocalGCN_DI) / (alpha + beta)
    fuse_DI = np.reshape(fuse_DI, (height * width, 1))
    threshold = otsu(fuse_DI)
    fuse_DI = np.reshape(fuse_DI, (height, width))
    bcm = np.zeros((height, width)).astype(np.uint8)
    bcm[fuse_DI > threshold] = 255
    bcm[fuse_DI <= threshold] = 0
    imageio.imsave('./result/Adaptive_Fuse.png', bcm)

    fuse_DI = 255 * ((fuse_DI - np.min(fuse_DI)) / (np.max(fuse_DI) - np.min(fuse_DI)))
    imageio.imsave('./result/Adaptive_Fuse_DI.png', fuse_DI.astype(np.uint8))

    ground_truth_changed = imageio.imread('./data/SG/GT.png')
    ground_truth_unchanged = 255 - ground_truth_changed
    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm)
    print(conf_mat)
    print(oa)
    print(f1)
    print(kappa_co)


if __name__ == '__main__':
    fuse_DI()
