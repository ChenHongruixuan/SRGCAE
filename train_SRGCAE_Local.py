import argparse
import time
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.segmentation import slic

from aux_func.acc_ass import assess_accuracy
from aux_func.clustering import otsu
from aux_func.graph_func import construct_affinity_matrix
from aux_func.preprocess import preprocess_img
from model.SRGCAE import GraphConvAutoEncoder_EdgeRecon


def load_checkpoint_for_evaluation(model, checkpoint):
    saved_state_dict = torch.load(checkpoint, map_location='cuda:0')
    model.load_state_dict(saved_state_dict)
    model.cuda()
    model.eval()


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

    img_t1 = preprocess_img(img_t1, d_type='sar', norm_type='stad')
    img_t2 = preprocess_img(img_t2, d_type='opt', norm_type='stad')
    # objects = np.load('./object_idx.npy')
    
    # Get actual segment IDs (SLIC may not start from 0 or may have gaps)
    unique_segments = np.unique(objects)
    obj_nums = len(unique_segments)

    node_set_t1 = []
    node_set_t2 = []
    for obj_idx in unique_segments:
        node_set_t1.append(img_t1[objects == obj_idx])
        node_set_t2.append(img_t2[objects == obj_idx])
    am_set_t1 = construct_affinity_matrix(img_t1, objects, args.band_width_t1)
    am_set_t2 = construct_affinity_matrix(img_t2, objects, args.band_width_t2)
    GCAE_model = GraphConvAutoEncoder_EdgeRecon(nfeat=3, nhid=16, nclass=3, dropout=0.5)
    optimizer = optim.AdamW(GCAE_model.parameters(), lr=1e-4, weight_decay=1e-6)
    GCAE_model.cuda()
    GCAE_model.train()

    # Edge information reconstruction
    for _epoch in range(args.epoch):
        for _iter in range(obj_nums):
            optimizer.zero_grad()
            node_t1 = node_set_t1[_iter]  # np.expand_dims(node_set_t1[_iter], axis=0)
            adj_t1, norm_adj_t1 = am_set_t1[_iter]  # np.expand_dims(am_set_t1[_iter], axis=0)
            node_t1 = torch.from_numpy(node_t1).cuda().float()
            adj_t1 = torch.from_numpy(adj_t1).cuda().float()
            norm_adj_t1 = torch.from_numpy(norm_adj_t1).cuda().float()

            node_t2 = node_set_t2[_iter]  # np.expand_dims(node_set_t2[_iter], axis=0)
            adj_t2, norm_adj_t2 = am_set_t2[_iter]  # np.expand_dims(am_set_t2[_iter], axis=0)
            node_t2 = torch.from_numpy(node_t2).cuda().float()
            adj_t2 = torch.from_numpy(adj_t2).cuda().float()
            norm_adj_t2 = torch.from_numpy(norm_adj_t2).cuda().float()

            feat_t1 = GCAE_model(node_t1, norm_adj_t1)
            feat_t2 = GCAE_model(node_t2, norm_adj_t2)

            recon_adj_t1 = torch.matmul(feat_t1, feat_t1.T)
            recon_adj_t2 = torch.matmul(feat_t2, feat_t2.T)

            cnstr_loss_t1 = F.mse_loss(input=recon_adj_t1, target=adj_t1) / adj_t1.size()[0]
            cnstr_loss_t2 = F.mse_loss(input=recon_adj_t2, target=adj_t2) / adj_t2.size()[0]
            ttl_loss = cnstr_loss_t2 + cnstr_loss_t1
            ttl_loss.backward()
            optimizer.step()
            if (_iter + 1) % 10 == 0:
                print(f'Epoch is {_epoch + 1}, iter is {_iter}, mse loss is {ttl_loss.item()}')
    torch.save(GCAE_model.state_dict(), './model_weight/' + str(time.time()) + '.pth')

    # Extracting deep edge representations & Change information mapping
    # Load pretrained weight
    # restore_from = './model_weight/SRGCAE_ER_SG.pth'
    # load_checkpoint_for_evaluation(GCAE_model, restore_from)
    GCAE_model.eval()
    diff_set = []

    for _iter in range(obj_nums):
        node_t1 = node_set_t1[_iter]  # np.expand_dims(node_set_t1[_iter], axis=0)
        node_t2 = node_set_t2[_iter]  # np.expand_dims(node_set_t2[_iter], axis=0)
        adj_t1, norm_adj_t1 = am_set_t1[_iter]  # np.expand_dims(am_set_t1[_iter], axis=0)
        adj_t2, norm_adj_t2 = am_set_t2[_iter]  # np.expand_dims(am_set_t2[_iter], axis=0)

        node_t1 = torch.from_numpy(node_t1).cuda().float()
        node_t2 = torch.from_numpy(node_t2).cuda().float()
        norm_adj_t1 = torch.from_numpy(norm_adj_t1).cuda().float()
        norm_adj_t2 = torch.from_numpy(norm_adj_t2).cuda().float()

        feat_t1 = GCAE_model(node_t1, norm_adj_t1)
        feat_t2 = GCAE_model(node_t2, norm_adj_t2)

        diff = torch.mean(torch.abs(feat_t1 - feat_t2))
        # diff = torch.sqrt(torch.sum(torch.square(feat_t1 - feat_t2), dim=1)) / norm_adj_t2.size()[0]
        diff_set.append(diff.data.cpu().numpy())

    diff_map = np.zeros((height, width))
    for i, obj_idx in enumerate(unique_segments):
        diff_map[objects == obj_idx] = diff_set[i]

    diff_map = np.reshape(diff_map, (height * width, 1))

    threshold = otsu(diff_map)
    diff_map = np.reshape(diff_map, (height, width))

    bcm = np.zeros((height, width)).astype(np.uint8)
    bcm[diff_map > threshold] = 255
    bcm[diff_map <= threshold] = 0

    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm)

    imageio.imsave('./result/SRGCAE_EdgeConc_' + str(time.time()) + '.png', bcm)
    diff_map = 255 * (diff_map - np.min(diff_map)) / (np.max(diff_map) - np.min(diff_map))
    imageio.imsave('./result/SRGCAE_EdgeConc_' + str(time.time()) + '_DI.png', diff_map.astype(np.uint8))

    print(conf_mat)
    print(oa)
    print(f1)
    print(kappa_co)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detecting land-cover changes on SG dataset")
    parser.add_argument('--n_seg', type=int, default=1500,
                        help='Approximate number of objects obtained by the segmentation algorithm')
    parser.add_argument('--cmp', type=int, default=5, help='Compectness of the obtained objects')
    parser.add_argument('--band_width_t1', type=float, default=0.4,
                        help='The bandwidth of the Gaussian kernel when calculating the adjacency matrix')
    parser.add_argument('--band_width_t2', type=float, default=0.5,
                        help='The bandwidth of the Gaussian kernel when calculating the adjacency matrix')
    parser.add_argument('--epoch', type=int, default=10, help='Training epoch of SRGCAE')
    args = parser.parse_args()

    train_model(args)
