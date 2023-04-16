import numpy as np


def preprocess_img(data, d_type, norm_type):
    pps_data = np.array(data).astype(np.float32)
    if d_type == 'opt':
        if norm_type == 'stad':
            pps_data = stad_img(pps_data, channel_first=False)
        else:
            pps_data = norm_img(pps_data, channel_first=False)
    elif d_type == 'sar':
        pps_data[np.abs(pps_data) <= 0] = np.min(pps_data[np.abs(pps_data) > 0])
        pps_data = np.log(pps_data + 1.0)
        if norm_type == 'stad':
            pps_data = stad_img(pps_data, channel_first=False)
            # sigma = np.std(pps_data)
            # mean = np.mean(pps_data)
            # idx_min = pps_data < (mean - 4 * sigma)
            # idx_max = pps_data > (mean + 4 * sigma)
            # pps_data[idx_min] = np.min(pps_data[~idx_min])
            # pps_data[idx_max] = np.max(pps_data[~idx_max])
        else:
            pps_data = norm_img(pps_data, channel_first=False)
            # sigma = np.std(pps_data)
            # mean = np.mean(pps_data)
            # idx_min = pps_data < (mean - 4* sigma)
            # idx_max = pps_data > (mean + 4 * sigma)
            # pps_data[idx_min] = np.min(pps_data[~idx_min])
            # pps_data[idx_max] = np.max(pps_data[~idx_max])
    return pps_data


def norm_img(img, channel_first):
    '''
        normalize value to [0, 1]
    '''
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        max_value = np.max(img, axis=1, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=1, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = (img - min_value) / diff_value
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (channel, height * width)
        max_value = np.max(img, axis=0, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=0, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = (img - min_value) / diff_value
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    return nm_img


def norm_img_2(img, channel_first):
    '''
    normalize value to [-1, 1]
    '''
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        max_value = np.max(img, axis=1, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=1, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = 2 * ((img - min_value) / diff_value - 0.5)
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (channel, height * width)
        max_value = np.max(img, axis=0, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=0, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = 2 * ((img - min_value) / diff_value - 0.5)
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    return nm_img


def stad_img(img, channel_first):
    """
    normalization image
    :param channel_first:
    :param img: (C, H, W)
    :return:
        norm_img: (C, H, W)
    """
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        mean = np.mean(img, axis=1, keepdims=True)  # (channel, 1)
        center = img - mean  # (channel, height * width)
        var = np.sum(np.power(center, 2), axis=1, keepdims=True) / (img_height * img_width)  # (channel, 1)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (height * width, channel)
        mean = np.mean(img, axis=0, keepdims=True)  # (1, channel)
        center = img - mean  # (height * width, channel)
        var = np.sum(np.power(center, 2), axis=0, keepdims=True) / (img_height * img_width)  # (1, channel)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    return nm_img
