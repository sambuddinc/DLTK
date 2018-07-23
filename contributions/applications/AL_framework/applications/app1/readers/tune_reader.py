from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import SimpleITK as sitk
import os
import numpy as np
import tensorflow as tf
import json

from dltk.io.augmentation import add_gaussian_noise, flip, extract_class_balanced_example_array
from dltk.io.preprocessing import whitening


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

    def get_config_for_app():
        dir_fn = os.path.dirname(__file__)[:-7]  # Remove 'readers' from filepath
        app_fn = os.path.join(dir_fn, 'app_config.json')
        with open(app_fn) as json_data:
            app_json = json.load(json_data)
        return app_json

    for f in file_references:

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Handle patches in here
            patch_id = f[0]
            patch_path = f[1]
            patch_postfix = '_patch.nii.gz'
            lbl_postfix = '_emseg.nii.gz'   # TODO: change to be manual annotation patch
            app_json = get_config_for_app()
            sitk_ref = None
            inputs_to_stack = []
            for i, input_type in enumerate(app_json['input_postfix']):
                # Read the image nii with sitk and keep the pointer to the sitk.Image of an input
                im_sitk = sitk.ReadImage(os.path.join(patch_path, str(str(patch_id) + input_type)))
                im = sitk.GetArrayFromImage(im_sitk)
                im = whitening(im)
                inputs_to_stack.append(im)
                if i == 0:
                    sitk_ref = im_sitk

            # Create a 4D multi-sequence image (i.e. [channels, x,y,z])
            # print("Correct stacking: ", len(inputs_to_stack) == len(app_json['input_postfix']))
            images = np.stack(inputs_to_stack, axis=-1).astype(np.float32)

            lbl_sitk = sitk.ReadImage(os.path.join(patch_path, str(str(patch_id) + lbl_postfix)))
            lbl = sitk.GetArrayFromImage(lbl_sitk).astype(np.int32)

            images, lbl = _augment(images, lbl)

            yield {'features': {'x': images},
                   'labels': {'y': lbl},
                   'sitk': sitk_ref}

        elif mode == tf.estimator.ModeKeys.EVAL:
            # Handle Eval stacks in here
            subject_id = f[0]
            subj_path = f[1]
            subj_prefix = f[2]
            stack_folder_path = subj_path

            app_json = get_config_for_app()
            sitk_ref = None
            inputs_to_stack = []
            for i, input_type in enumerate(app_json['input_postfix']):
                # Read the image nii with sitk and keep the pointer to the sitk.Image of an input
                im_sitk = sitk.ReadImage(os.path.join(stack_folder_path, str(subj_prefix + input_type)))
                im = sitk.GetArrayFromImage(im_sitk)
                im = whitening(im)
                inputs_to_stack.append(im)
                if i == 0:
                    sitk_ref = im_sitk

            # Create a 4D multi-sequence image (i.e. [channels, x,y,z])
            # print("Correct stacking: ", len(inputs_to_stack) == len(app_json['input_postfix']))
            images = np.stack(inputs_to_stack, axis=-1).astype(np.float32)

            lbl_sitk = sitk.ReadImage(os.path.join(stack_folder_path, str(subj_prefix + app_json['output_postfix'])))
            lbl = sitk.GetArrayFromImage(lbl_sitk).astype(np.int32)

            # Remove other class labels to leave just the grey matter
            lbl[lbl != 2.] = 0.
            lbl[lbl == 2.] = 1.

            # Check if reader is returning training examples or full images
            if params['extract_examples']:
                # print("extracting training examples (not full images)")
                n_examples = params['n_examples']
                example_size = params['example_size']

                images, lbl = extract_class_balanced_example_array(
                    image=images,
                    label=lbl,
                    example_size=example_size,
                    n_examples=n_examples,
                    classes=app_json['num_classes'])

                assert not np.any(np.isnan(images))
                for e in range(n_examples):
                    yield {'features': {'x': images[e].astype(np.float32)},
                           'labels': {'y': lbl[e].astype(np.int32)},
                           'subject_id': subject_id}
            else:
                assert not np.any(np.isnan(images))
                assert sitk_ref is not None
                yield {'features': {'x': images},
                       'labels': {'y': lbl},
                       'sitk': sitk_ref,
                       'subject_id': subject_id,
                       'path': subj_path,
                       'prefix': subj_prefix}

    return
