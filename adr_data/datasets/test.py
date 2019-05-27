from adr_data.datasets.svhn import load_svhn
from adr_data.datasets.mnist import load_mnist
# import matplotlib.pyplot as plt
import numpy as np

def return_dataset(data, scale=False, usps=False, all_use=False):
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist(scale=scale, usps=usps, all_use=all_use)
        print(train_image.shape)
    # if data == 'usps':
    #     train_image, train_label, \
    #     test_image, test_label = load_usps(all_use=all_use)
    return train_image, train_label, test_image, test_label


def dataset_read(source, target, pixel_norm=True, scale=False, all_use=False):
    S = {}
    S_test = {}
    T = {}
    T_test = {}
    usps = False
    if source == 'usps' or target == 'usps':
        usps = True

    train_source, s_label_train, test_source, s_label_test = return_dataset(source, scale=scale,
                                                                            usps=usps, all_use=all_use)
    train_target, t_label_train, test_target, t_label_test = return_dataset(target, scale=scale, usps=usps,
                                                                            all_use=all_use)
    #normalize with mean value of pixels
    if pixel_norm:
        pixel_mean = np.vstack([train_source, train_target]).mean((0,))
        train_source = (train_source - pixel_mean) / float(255)
        test_source = (test_source - pixel_mean) / float(255)
        train_target = (train_target - pixel_mean) / float(255)
        test_target = (test_target - pixel_mean) / float(255)

    return train_source, test_source, train_target, test_target, s_label_train, s_label_test, t_label_train, t_label_test


# train_source, test_source, train_target, test_target, s_label_train, s_label_test, t_label_train, t_label_test = dataset_read(
# source='svhn', target='mnist', pixel_norm=True, scale=True, all_use=False
# )

# print(train_source.shape)
# print(np.max(train_source))
# print(np.min(train_source))
# print(test_target.shape)
# print(np.max(test_target))
# print(np.min(test_target))
# print(np.max(train_target))
# print(np.min(train_target))
# print(np.max(test_source))
# print(np.min(test_source))


# train_source = np.transpose(train_source,[0,2,3,1])
# test_source = np.transpose(test_source,[0,2,3,1])
# train_target = np.transpose(train_target,[0,2,3,1])
# test_target = np.transpose(test_target,[0,2,3,1])
#
# for i in range(20):
#     plt.imshow(train_source[i])
#     plt.title(s_label_test[i])
#     plt.show()