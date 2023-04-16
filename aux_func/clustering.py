import numpy as np


def otsu(data, num=1000):
    max_value = np.max(data)
    min_value = np.min(data)

    total_num = data.shape[0]
    step_value = (max_value - min_value) / num
    value = min_value + step_value
    best_threshold = min_value
    best_inter_class_var = 0
    while value <= max_value:
        data_1 = data[data < value]
        data_2 = data[data >= value]
        w1 = data_1.shape[0] / total_num
        w2 = data_2.shape[0] / total_num

        mean_1 = data_1.mean()
        mean_2 = data_2.mean()

        inter_class_var = w1 * w2 * np.power((mean_1 - mean_2), 2)
        if best_inter_class_var < inter_class_var:
            best_inter_class_var = inter_class_var
            best_threshold = value
        value += step_value
    # bcm = np.zeros(data.shape).astype(np.uint8)
    # bcm[data <= best_threshold] = 0
    # bcm[data > best_threshold] = 255
    return best_threshold
