from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import SimpleITK as sitk
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dltk.io.augmentation import add_gaussian_noise, flip, extract_class_balanced_example_array, \
    extract_random_example_array
from dltk.io.preprocessing import whitening

t2_postfix = "T2w_restore_brain.nii.gz"
label_postfix = "drawem_tissue_labels.nii.gz"
NUM_CLASSES = 2

def read_fn(file_references, mode, params=None):
    """A custom python read function for interfacing with nii image files.

        Args:
            file_references (list): A list of lists containing file references, such
                as [['id_0', 'image_filename_0', target_value_0], ...,
                ['id_N', 'image_filename_N', target_value_N]].
            mode (str): One of the tf.estimator.ModeKeys strings: TRAIN, EVAL or
                PREDICT.
            params (dict, optional): A dictionary to parameterise read_fn ouputs
                (e.g. reader_params = {'n_examples': 10, 'example_size':
                [64, 64, 64], 'extract_examples': True}, etc.).

        Yields:
            dict: A dictionary of reader outputs for dltk.io.abstract_reader.
        """

    def _augment(img, lbl):
        """An image augmentation function"""
        img = add_gaussian_noise(img, sigma=0.1)
        [img, lbl] = flip([img, lbl], axis=1)
        return img, lbl

    for f in file_references:
        subject_id = f[0]
        img_path = f[1]
        img_prefix = f[2]

        # Read the image nii with sitk and keep the pointer to the sitk.Image of an input
        # print(os.getcwd())
        t2_sitk = sitk.ReadImage(str(img_path + img_prefix + t2_postfix))
        t2 = sitk.GetArrayFromImage(t2_sitk)

        # Normalise volume images
        t2 = whitening(t2)

        # Create a 4D multi-sequence image (i.e. [channels, x,y,z])
        images = np.stack([t2], axis=1).astype(np.float32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            print("Predict not yet implemented, please try a different mode")
            yield {'features': {'x': images},
                   'labels': None,
                   'sitk': t2_sitk,
                   'subject_id': subject_id}

        lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path + img_prefix + label_postfix))).astype(
            np.int32)
        #print(lbl.shape)
        # Remove other class labels to leave just the grey matter
        #non_cortical_indices = lbl != 2
        lbl[lbl != 1.] = 0.
        #print(lbl.shape)
        # Augment if in training
        if mode == tf.estimator.ModeKeys.TRAIN:
            images, lbl = _augment(images, lbl)

        # Check if reader is returning training examples or full images
        if params['extract_examples']:
            # print("extracting training examples (not full images)")
            n_examples = params['n_examples']
            example_size = params['example_size']

            images = images.reshape([lbl.shape[0], lbl.shape[1], lbl.shape[2], 1])

            images, lbl = extract_class_balanced_example_array(
                image=images,
                label=lbl,
                example_size=example_size,
                n_examples=n_examples,
                classes=NUM_CLASSES)
            # examples = extract_random_example_array([images, lbl],
            #                                         example_size=example_size,
            #                                         n_examples=n_examples)
            # images = examples[0]
            # lbl = examples[1]

            for e in range(n_examples):
                yield {'features': {'x': images[e].astype(np.float32)},
                       'labels': {'y': lbl[e].astype(np.int32)},
                       'subject_id': subject_id}
        else:
            print("extracting full images (not training examples)")
            images = images.reshape([lbl.shape[0],lbl.shape[1], lbl.shape[2], 1])
            yield {'features': {'x': images},
                   'labels': {'y': lbl},
                   'sitk': t2_sitk,
                   'subject_id': subject_id}

    return
