import numpy as np


def assess_accuracy(gt_changed, gt_unchanged, changed_map):
    """
    assess accuracy of changed map based on ground truth
    :param gt_changed: changed ground truth
    :param gt_unchanged: unchanged ground truth
    :param changed_map: changed map
    :return: confusion matrix and overall accuracy
    """
    change_index = (gt_changed == 255)
    unchanged_index = (gt_unchanged == 255)
    n_cc = (changed_map[change_index] == 255).sum()  # changed-changed
    n_uu = (changed_map[unchanged_index] == 0).sum()  # unchanged-unchanged
    n_cu = change_index.sum() - n_cc  # changed-unchanged
    n_uc = unchanged_index.sum() - n_uu  # unchanged-changed

    conf_mat = np.array([[n_cc, n_cu], [n_uc, n_uu]])

    pre = n_cc / (n_cc + n_uc)
    rec = n_cc / (n_cc + n_cu)
    f1 = 2 * pre * rec / (pre + rec)

    over_acc = (conf_mat.diagonal().sum()) / (conf_mat.sum())
    # pe = np.array(0, np.int64)
    pe = ((n_cc + n_cu) / conf_mat.sum() * (n_cc + n_uc) + (n_uu + n_uc) / conf_mat.sum() * (
            n_uu + n_cu)) / conf_mat.sum()
    kappa_co = (over_acc - pe) / (1 - pe)
    return conf_mat, over_acc, f1, kappa_co


def assess_accuracy_from_conf_mat(conf_mat):
    """
    assess accuracy of changed map based on ground truth
    :param gt_changed: changed ground truth
    :param gt_unchanged: unchanged ground truth
    :param changed_map: changed map
    :return: confusion matrix and overall accuracy
    """

    n_cc = conf_mat[0, 0]
    n_cu = conf_mat[0, 1]
    n_uc = conf_mat[1, 0]
    n_uu = conf_mat[1, 1]
    # conf_mat = np.array([[n_cc, n_cu], [n_uc, n_uu]])

    pre = n_cc / (n_cc + n_uc)
    rec = n_cc / (n_cc + n_cu)
    f1 = 2 * pre * rec / (pre + rec)

    over_acc = (conf_mat.diagonal().sum()) / (conf_mat.sum())
    # pe = np.array(0, np.int64)
    pe = ((n_cc + n_cu) / conf_mat.sum() * (n_cc + n_uc) + (n_uu + n_uc) / conf_mat.sum() * (
            n_uu + n_cu)) / conf_mat.sum()
    kappa_co = (over_acc - pe) / (1 - pe)
    return conf_mat, over_acc, f1, kappa_co, kappa_co
