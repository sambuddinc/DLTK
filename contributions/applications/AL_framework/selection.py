from __future__ import division
import numpy as np
import random
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

def small_f(S_a, I_x):

    # return the sim of the image in S_a with highest sim with I_x
    max_sim = 0
    max_sim_idx = 0
#     print(I_sa.shape)
#     print(I_x.shape)
    for I_sa in S_a:
        sim = cos_sim(I_sa, I_x)
#         print(sim.shape)
        if sim > max_sim:
            max_sim = sim

    return max_sim


def big_f(S_a, S_u):
    # Sum small_f for all images in S_u f_small(S_a, I_from_S_u)
    current_sum = 0
    for I_su in S_u:
        current_sum += small_f(S_a, I_su)

    return current_sum


def cos_sim(I_i, I_j):
    return np.dot(I_i, I_j.T) / I_i.shape[0] ** 2


def max_rep(S_u, S_c, Sc_idx):
    S_a_idx = []
    S_a = []

    while len(S_a) < small_k:
        current_best = 0
        current_best_idx = None
        for i, img_and_idx in enumerate(zip(S_c, Sc_idx)):
            S_a.append(img_and_idx[0])
            tmp_score = big_f(S_a, S_u)
            if tmp_score > current_best:
                current_best = tmp_score
                current_best_idx = i

            S_a.pop()

        S_a.append(S_c[current_best_idx])
        S_a_idx.append(Sc_idx[current_best_idx])
        S_c.pop(current_best_idx)
        Sc_idx.pop(current_best_idx)

    return S_a_idx

# With our data ready, we will extract random patches (in the same place for each of the three data points per subject)

def extract_random_patches(image_list, conf_list, feat_list,
                                 example_size=[1, 64, 64],
                                 n_examples=16):
    """Randomly extract training examples from image (and a corresponding label).
        Returns an image example array and the corresponding label array.

    Args:
        image_list (np.ndarray or list or tuple): image(s) to extract random
            patches from
        example_size (list or tuple): shape of the patches to extract
        n_examples (int): number of patches to extract in total

    Returns:
        np.ndarray, np.ndarray: class-balanced patches extracted from full
        images with the shape [batch, example_size..., image_channels]
    """

    assert n_examples > 0

    was_singular = False
    if isinstance(image_list, np.ndarray):
        print("was singular")
        image_list = [image_list]
        conf_list = [conf_list]
        feat_list = [feat_list]
        was_singular = True

    assert all([i_s >= e_s for i_s, e_s in zip(image_list[0].shape, example_size)]), \
        'Image must be bigger than example shape'
    assert (image_list[0].ndim - 1 == len(example_size)
            or image_list[0].ndim == len(example_size)), \
        'Example size doesnt fit image size'

    for i in image_list:
        if len(image_list) > 1:
            assert (i.ndim - 1 == image_list[0].ndim
                    or i.ndim == image_list[0].ndim
                    or i.ndim + 1 == image_list[0].ndim), \
                'Example size doesnt fit image size'

            assert all([i0_s == i_s for i0_s, i_s in zip(image_list[0].shape, i.shape)]), \
                'Image shapes must match'

    rank = len(example_size)

    # Extract random examples from image and label
    valid_loc_range = [image_list[0].shape[i] - example_size[i] for i in range(rank)]

    rnd_loc = [np.random.randint(valid_loc_range[dim], size=n_examples)
               if valid_loc_range[dim] > 0
               else np.zeros(n_examples, dtype=int) for dim in range(rank)]

    examples = [[]] * len(image_list)
    examples_f = [[]] * len(feat_list)
    examples_c = [[]] * len(conf_list)
    for i in range(n_examples):
        slicer = [slice(rnd_loc[dim][i], rnd_loc[dim][i] + example_size[dim])
                  for dim in range(rank)]

        for j in range(len(image_list)):
            ex_image = image_list[j][slicer][np.newaxis]
            ex_conf = conf_list[j][slicer][np.newaxis]
            ex_feat = feat_list[j][slicer][np.newaxis]
            # Concatenate and return the examples
            examples[j] = np.concatenate((examples[j], ex_image), axis=0) \
                if (len(examples[j]) != 0) else ex_image
            examples_f[j] = np.concatenate((examples_f[j], ex_feat), axis=0) \
                if (len(examples_f[j]) != 0) else ex_feat
            examples_c[j] = np.concatenate((examples_c[j], ex_conf), axis=0) \
                if (len(examples_c[j]) != 0) else ex_conf

    if was_singular:
        return examples[0]
    return [examples, examples_c, examples_f]


# Load our example data into memory
num_images = 5

data_dir = "/Users/sambudd2/PycharmProjects/DLTK/contributions/applications/AL_framework/datasets/ALout/"

file_names = pd.read_csv(
                data_dir + "im_refs.csv",
                dtype=object,
                keep_default_na=False,
                na_values=[]).as_matrix()



images = []
confidences = []
features = []

for im in file_names:
    im_id = im[0]
#     print(im_id)
    im_name = im[2] + "T2w_restore_brain.nii.gz"
    image = sitk.GetArrayFromImage(sitk.ReadImage(data_dir+im_name))
    conf = sitk.GetArrayFromImage(sitk.ReadImage(data_dir+str(im_id)+"_conf.nii.gz"))
    feat = sitk.GetArrayFromImage(sitk.ReadImage(data_dir+str(im_id)+"_feat.nii.gz"))
    patches = extract_random_patches(image, conf, feat, n_examples=100)
    images.append(patches[0])
    features.append(patches[1])
    confidences.append(patches[2])

# We now have a list of patches: lets vis some!
# ex_i = np.array(ex_i)
# ex_c = np.array(ex_c)
# ex_f = np.array(ex_f)
print(ex_i.shape)
print(ex_c.shape)
print(ex_f.shape)
fig, axes = plt.subplots(10, 3, figsize=(30,30))
for i in range(0, len(axes)):
    axes[i, 0].imshow(ex_i[i, 0])
    axes[i, 1].imshow(ex_c[i, 0])
    axes[i, 2].imshow(ex_f[i, 0])