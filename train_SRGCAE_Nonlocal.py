import argparse
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.segmentation import slic
from sklearn.metrics.pairwise import pairwise_distances

from aux_func.acc_ass import assess_accuracy
from aux_func.clustering import otsu
from aux_func.graph_func import construct_affinity_matrix
from aux_func.preprocess import preprocess_img
from model.SRGCAE import GraphConvAutoEncoder_VertexRecon


def load_checkpoint_for_evaluation(model, checkpoint):
    saved_state_dict = torch.load(checkpoint, map_location='cuda:0')
    model.load_state_dict(saved_state_dict)
    model.cuda()
    model.eval()


def cal_nonlocal_dist(vector, band_width):
    euc_dis = pairwise_distances(vector)
    gaus_dis = np.exp(- euc_dis * euc_dis / (band_width * band_width))
    return gaus_dis


def train_model(args):
    img_t1 = imageio.imread('./data/SG/T1.png')  # .astype(np.float32)
    img_t2 = imageio.imread('./data/SG/T2.png')  # .astype(np.float32)
    ground_truth_changed = imageio.imread('./data/SG/GT.png')
    ground_truth_unchanged = 255 - ground_truth_changed

    height, width, channel_t1 = img_t1.shape
    _, _, channel_t2 = img_t2.shape

    # In our paper, the object map is obtained through FNEA algorithm based on eCognition.
    # According to our response to reviewers, SLIC can be implemented in Python as an alternative algorithm.
    # There is little difference in accuracy between the results obtained by these two methods.
    # Yet SLIC can only process images in three bands, so you will need to process images in more than three bands.
    objects = slic(img_t2, n_segments=args.n_seg, compactness=args.cmp)

    img_t1 = preprocess_img(img_t1, d_type='sar', norm_type='norm')
    img_t2 = preprocess_img(img_t2, d_type='opt', norm_type='norm')

    obj_nums = np.max(objects) + 1

    node_set_t1 = []
    node_set_t2 = []
    for obj_idx in range(obj_nums):
        node_set_t1.append(img_t1[objects == obj_idx])
        node_set_t2.append(img_t2[objects == obj_idx])
    am_set_t1 = construct_affinity_matrix(img_t1, objects, args.band_width_t1)
    am_set_t2 = construct_affinity_matrix(img_t2, objects, args.band_width_t2)

    GCAE_model = GraphConvAutoEncoder_VertexRecon(nfeat=3, nhid=16, nclass=3, dropout=0.5)
    optimizer = optim.Adam(GCAE_model.parameters(), lr=1e-4, weight_decay=1e-4)
    GCAE_model.cuda()
    GCAE_model.train()

    # Vertex information reconstruction
    for _epoch in range(args.epoch):
        for _iter in range(obj_nums):
            optimizer.zero_grad()
            node_t1 = node_set_t1[_iter]  # np.expand_dims(node_set_t1[_iter], axis=0)
            node_t2 = node_set_t2[_iter]  # np.expand_dims(node_set_t2[_iter], axis=0)
            _, norm_adj_t1 = am_set_t1[_iter]  # np.expand_dims(am_set_t1[_iter], axis=0)
            _, norm_adj_t2 = am_set_t2[_iter]  # np.expand_dims(am_set_t2[_iter], axis=0)

            node_t1 = torch.from_numpy(node_t1).cuda().float()
            node_t2 = torch.from_numpy(node_t2).cuda().float()

            norm_adj_t1 = torch.from_numpy(norm_adj_t1).cuda().float()
            norm_adj_t2 = torch.from_numpy(norm_adj_t2).cuda().float()

            cstr_node_t1, feat_t1 = GCAE_model(node_t1, norm_adj_t1)
            cstr_node_t2, feat_t2 = GCAE_model(node_t2, norm_adj_t2)

            cnstr_loss_t1 = F.mse_loss(input=cstr_node_t1, target=node_t1)
            cnstr_loss_t2 = F.mse_loss(input=cstr_node_t2, target=node_t2)
            ttl_loss = cnstr_loss_t2 + cnstr_loss_t1
            ttl_loss.backward()
            optimizer.step()
            if (_iter + 1) % 10 == 0:
                print(f'Epoch is {_epoch + 1}, iter is {_iter}, mse loss is {ttl_loss.item()}')
    # torch.save(GCAE_model.state_dict(), './model_weight/' + str(time.time()) + '.pth')
    # Extracting deep vertex representations
    # Load pretrained weight
    # restore_from = './model_weight/SRGCAE_VR_SG.pth'
    # load_checkpoint_for_evaluation(GCAE_model, restore_from)
    GCAE_model.eval()
    feat_set_t1 = []
    feat_set_t2 = []

    for _iter in range(obj_nums):
        node_t1 = node_set_t1[_iter]
        node_t2 = node_set_t2[_iter]
        _, norm_adj_t1 = am_set_t1[_iter]
        _, norm_adj_t2 = am_set_t2[_iter]

        node_t1 = torch.from_numpy(node_t1).cuda().float()
        node_t2 = torch.from_numpy(node_t2).cuda().float()
        norm_adj_t1 = torch.from_numpy(norm_adj_t1).cuda().float()
        norm_adj_t2 = torch.from_numpy(norm_adj_t2).cuda().float()

        _, feat_t1 = GCAE_model(node_t1, norm_adj_t1)
        _, feat_t2 = GCAE_model(node_t2, norm_adj_t2)

        feat_t1 = torch.mean(feat_t1, dim=0)
        feat_t2 = torch.mean(feat_t2, dim=0)
        feat_set_t1.append(feat_t1.data.cpu().numpy())
        feat_set_t2.append(feat_t2.data.cpu().numpy())

    feat_set_t1 = np.array(feat_set_t1)
    feat_set_t2 = np.array(feat_set_t2)

    dist_set_t1 = cal_nonlocal_dist(feat_set_t1, args.deep_band_width_t1)
    dist_set_t2 = cal_nonlocal_dist(feat_set_t2, args.deep_band_width_t2)

    neigh_idx_t1 = np.argsort(-dist_set_t1, axis=1)
    neigh_idx_t2 = np.argsort(-dist_set_t2, axis=1)

    fx_node_dist = np.zeros((obj_nums, 1))
    fy_node_dist = np.zeros((obj_nums, 1))

    # Change information mapping
    for i in range(obj_nums):
        fx_node_dist[i] = np.mean(
            np.abs(dist_set_t1[i, neigh_idx_t1[i, 1:args.knn_num]] - dist_set_t1[i, neigh_idx_t2[i, 1:args.knn_num]]))
        fy_node_dist[i] = np.mean(
            np.abs(dist_set_t2[i, neigh_idx_t2[i, 1:args.knn_num]] - dist_set_t2[i, neigh_idx_t1[i, 1:args.knn_num]]))
    diff_map = np.zeros((height, width))

    for i in range(0, obj_nums):
        diff_map[objects == i] = fx_node_dist[i] + fy_node_dist[i]
    diff_map = np.reshape(diff_map, (height * width, 1))
    threshold = otsu(diff_map)
    diff_map = np.reshape(diff_map, (height, width))
    bcm = np.zeros((height, width)).astype(np.uint8)
    bcm[diff_map > threshold] = 255
    bcm[diff_map <= threshold] = 0

    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm)

    imageio.imsave('./result/SRGCAE_VerConc_' + str(time.time()) + '.png', bcm)
    diff_map = 255 * (diff_map - np.min(diff_map)) / (np.max(diff_map) - np.min(diff_map))
    imageio.imsave('./result/SRGCAE_VerConc_' + str(time.time()) + '_DI.png', diff_map.astype(np.uint8))

    print(conf_mat)
    print(oa)
    print(f1)
    print(kappa_co)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detecting land-cover changes on SG dataset")
    parser.add_argument('--n_seg', type=int, default=5000,
                        help='Approximate number of objects obtained by the segmentation algorithm')
    parser.add_argument('--cmp', type=int, default=5, help='Compectness of the obtained objects')
    parser.add_argument('--band_width_t1', type=float, default=1,
                        help='The bandwidth of the Gaussian kernel when calculating the adjacency matrix')
    parser.add_argument('--band_width_t2', type=float, default=0.7,
                        help='The bandwidth of the Gaussian kernel when calculating the adjacency matrix')
    parser.add_argument('--deep_band_width_t1', type=float, default=0.15,
                        help='The bandwidth of the Gaussian kernel when calculating the adjacency matrix using deep vertex representations')
    parser.add_argument('--deep_band_width_t2', type=float, default=0.15,
                        help='The bandwidth of the Gaussian kernel when calculating the adjacency matrix using deep vertex representations')
    parser.add_argument('--knn_num', type=int, default=100,
                        help='the number of most similar objects for calculating nonlocal structural relationship')
    parser.add_argument('--epoch', type=int, default=15, help='Training epoch of SRGCAE')
    args = parser.parse_args()

    train_model(args)
