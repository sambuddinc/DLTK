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
# t1_postfix = "T1w_restore_brain.nii.gz"

NUM_CHANNELS = 1


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
        slice_index = int(f[3])
        man_path = f[4]
        img_path = f[5]
        img_prefix = f[6]

        # Read the image nii with sitk and keep the pointer to the sitk.Image of an input
        # print(os.getcwd())
        t2_sitk = sitk.ReadImage(str(img_path + img_prefix + t2_postfix))
        t2 = sitk.GetArrayFromImage(t2_sitk)

        # Drop all unannotated slices
        t2 = t2[slice_index, :, :]

        # Normalise volume images
        t2 = whitening(t2)

        # Read t1 image
        # t1_sitk = sitk.ReadImage(str(img_path + img_prefix + t1_postfix))
        # t1 = sitk.GetArrayFromImage(t1_sitk)
        #
        # # Drop all unannotated slices
        # t1 = t1[slice_index, :, :]
        #
        # t1 = whitening(t1)

        # Create a 4D multi-sequence image (i.e. [channels, x,y,z])
        images = np.stack([t2], axis=1).astype(np.float32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            images.reshape([1, 290, 290, NUM_CHANNELS]) #TODO: Remove magic numbers, check what calls this?
            print("Predict not yet implemented, please try a different mode")
            yield {'features': {'x': images},
                   'labels': None,
                   'sitk': t2_sitk,
                   'subject_id': subject_id}

        lbl = sitk.GetArrayFromImage(sitk.ReadImage(man_path)).astype(
            np.int32)

        # Drop unnanotated slices

        lbl = lbl[slice_index, :, :]

        # Augment if in training
        if mode == tf.estimator.ModeKeys.TRAIN:
            images, lbl = _augment(images, lbl)

        # Check if reader is returning training examples or full images
        if params['extract_examples']:
            # print("extracting training examples (not full images)")
            n_examples = params['n_examples']
            example_size = params['example_size']
            #x = lbl.shape[0]
            #y = lbl.shape[1]
            #images = images.reshape([NUM_CHANNELS, x,y])
            #lbl = lbl.reshape([NUM_CHANNELS, x,y])
            lbl = lbl.reshape([1, lbl.shape[0], lbl.shape[1]])
            images = images.reshape([lbl.shape[0], lbl.shape[1], lbl.shape[2], NUM_CHANNELS])

            #print(images.shape)
            #print(lbl.shape)
            #print(example_size)
            images, lbl = extract_class_balanced_example_array(
                image=images,
                label=lbl,
                example_size=example_size,
                n_examples=n_examples,
                classes=2)
            #print(images.shape)
            #print(example_size)
            #examples = extract_random_example_array([images, lbl],
             #                                        example_size=example_size,
              #                                       n_examples=n_examples)
            #images = examples[0]
            #lbl = examples[1]

            for e in range(n_examples):
                #print(images[e].shape)
                yield {'features': {'x': images[e].astype(np.float32)},
                       'labels': {'y': lbl[e].astype(np.int32)},
                       'subject_id': subject_id}
        else:
            #images = images.reshape([lbl.shape[0], lbl.shape[1], lbl.shape[2], NUM_CHANNELS])
            print("extracting full images (not training examples)")
            lbl = lbl.reshape([1, lbl.shape[0], lbl.shape[1]])
            images = images.reshape([lbl.shape[0], lbl.shape[1], lbl.shape[2], NUM_CHANNELS])
            yield {'features': {'x': images},
                   'labels': {'y': lbl},
                   'sitk': t2_sitk,
                   'subject_id': subject_id,
                   'slice_index': slice_index}

    return
