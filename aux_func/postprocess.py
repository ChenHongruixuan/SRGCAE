import cv2 as cv
import imageio
from acc_ass import assess_accuracy


def postprocess(data):
    kernel_1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (45, 45))
    kernel_2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))

    bcm_2 = cv.morphologyEx(data, cv.MORPH_CLOSE, kernel_1, 1)
    bcm_2 = cv.morphologyEx(bcm_2, cv.MORPH_OPEN, kernel_2, 1)

    imageio.imsave('../result/Adaptive_Fuse_pps.png', bcm_2)

    ground_truth_changed = imageio.imread('../data/SG/GT.png')
    ground_truth_unchanged = 255 - ground_truth_changed
    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm_2)
    print(conf_mat)
    print(oa)
    print(f1)
    print(kappa_co)


if __name__ == '__main__':
    cm_path = '../result/Adaptive_Fuse.png'
    cm = imageio.imread(cm_path)
    postprocess(cm)
