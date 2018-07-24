from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import SimpleITK as sitk
import random
import argparse
import os
import tensorflow as tf
import json
from importlib import import_module
import threading

from tensorflow.contrib import predictor

from dltk.utils import sliding_window_segmentation_inference


def select_patches_func():
    # Start app initialisation

    # Load app config -> check model status before initialising
    app_json = get_config_for_app()

    model_status = int(app_json['model_status'])
    new_model_status = -1
    if model_status == 0:
        # no model exists yet: init the model and begin training in new thread.
        new_model_status = 0
    elif model_status == 1:
        # model is under training -> do nothing
        new_model_status = 1
    elif model_status == 2:
        # model is ready -> DO THE THANG
        new_model_status = 2
        patch_select_thread = threading.Thread(target=select_patches(), args={})
        patch_select_thread.start()
        print('patch select thread started, should return now')

    elif model_status == 3:
        # model and patches are ready -> return the patches
        new_model_status = 3

    return new_model_status


def select_patches():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='dhcp brain segmentation deploy')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p',
                        default=os.path.join(os.path.dirname(__file__), 'model'))

    parser.add_argument('--csv',
                        default=os.path.join(os.path.dirname(__file__), 'data', 'subject_data.csv'))

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Call training
    predict(args)
    return


def get_config_for_app():
    app_fn = os.path.join(os.path.dirname(__file__), 'app_config.json')
    with open(app_fn) as json_data:
        app_json = json.load(json_data)
    return app_json


def write_app_config(app_json):
    app_fn = os.path.join(os.path.dirname(__file__), 'app_config.json')
    with open(app_fn, 'w') as outfile:
        json.dump(app_json, outfile, indent=4)


def predict(args):
    file_names = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()

    # We want to predict only on unannotated subjects
    ua_fn = []
    for i, row in enumerate(file_names):
        if row[6] == '1':
            ua_fn.append(row)
    print('selecting from', len(ua_fn), 'image stacks')
    # From the model_path, parse the latest saved model and restore a
    # predictor from it
    export_dir = [os.path.join(args.model_path, o) for o in os.listdir(args.model_path)
                  if os.path.isdir(os.path.join(args.model_path, o)) and
                  o.isdigit()][-1]
    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)

    # Fetch the output probability op of the trained network
    y_prob = my_predictor._fetch_tensors['y_prob']
    num_classes = y_prob.get_shape().as_list()[-1]

    # EDIT: Fetch the feature vector op of the trained network
    logits = my_predictor._fetch_tensors['logits']

    print("Preparing to predict")
    # Iterate through the files, predict on the full volumes and compute a Dice
    # coefficient
    app_json = get_config_for_app()
    # module_name = 'contributions.applications.AL_framework.applications.app' + str(app_json['id']) + '.readers.'
    module_name = 'readers.'
    if app_json['reader_type'] == "Patch":
        module_name = module_name + 'patch_reader'
    elif app_json['reader_type'] == "Slice":
        module_name = module_name + 'slice_reader'
    elif app_json['reader_type'] == "Stack":
        module_name = module_name + 'stack_reader'
    else:
        print("Unsupported reader type: please specify a new one")
        return

    # mod = import_module('readers.stack_reader')
    mod = import_module(module_name)
    read_fn = vars(mod)['read_fn']
    reader_params = {'extract_examples': False}
    for output in read_fn(file_references=ua_fn,
                          mode=tf.estimator.ModeKeys.EVAL,
                          params=reader_params):
        print("Predicting on an entry")
        # Parse the read function output and add a dummy batch dimension as
        # required
        img = np.expand_dims(output['features']['x'], axis=0)

        # Do a sliding window inference with our DLTK wrapper
        pred = sliding_window_segmentation_inference(
            session=my_predictor.session,
            ops_list=[y_prob],
            sample_dict={my_predictor._feed_tensors['x']: img},
            batch_size=app_json['batch_size'])[0]

        print("Prediction: " + str(pred.shape))

        features = sliding_window_segmentation_inference(
            session=my_predictor.session,
            ops_list=[logits],
            sample_dict={my_predictor._feed_tensors['x']: img},
            batch_size=app_json['batch_size'])[0]

        class_confidences = pred

        # Calculate the prediction from the probabilities
        pred = np.argmax(pred, -1)

        # Save the file as .nii.gz using the header information from the
        # original sitk
        # print(output)
        subj_path = output['path']
        output_fn = os.path.join(subj_path, '{}bronze_seg.nii.gz'.format(output['prefix']))
        new_sitk = sitk.GetImageFromArray(pred[0].astype(np.int32))
        print('pred unique:', np.unique(np.array(pred[0]).flatten()))
        new_sitk.CopyInformation(output['sitk'])
        sitk.WriteImage(new_sitk, output_fn)

        # Save the feature vector file as a .nii.gz using header info from origincal sitk
        print("Features: " + str(features.shape))
        feature_sitk = sitk.GetImageFromArray(features[0])
        feature_sitk.CopyInformation(output['sitk'])
        sitk.WriteImage(feature_sitk, os.path.join(subj_path, '{}feat.nii.gz'.format(output['prefix'])))

        # Save the confidence vector file as a .nii.gz using header info from original stack
        print("Confidences: " + str(class_confidences.shape))
        print(np.unique(np.array(class_confidences).flatten()))
        conf_sitk = sitk.GetImageFromArray(class_confidences[0])
        conf_sitk.CopyInformation(output['sitk'])
        sitk.WriteImage(conf_sitk, os.path.join(subj_path, '{}conf.nii.gz'.format(output['prefix'])))

    # Now perform patch selection with the saved outputs
    select_patch_batch(args, app_json)


def select_patch_batch(args, app_json):
    images = [[],[]]
    confidences = []
    features = []
    segs = []
    em_segs = []
    file_names = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()

    # We want to predict only on unannotated subjects
    ua_fn = []
    for i, row in enumerate(file_names):
        if row[6] == '1':
            ua_fn.append(row)

    patch_count = 0
    for im in ua_fn:
        im_id = im[0]
        im_fn = os.path.join(im[1])
        im_pref = im[2]
        ims = []
        for i, im in enumerate(app_json['input_postfix']):
            ima = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(im_fn, im_pref + str(im))))
            ims.append(ima)
        image = np.stack(ims, axis=-1)
        conf = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(im_fn, str(im_pref) +"conf.nii.gz")))
        feat = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(im_fn, str(im_pref) +"feat.nii.gz")))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(im_fn, str(im_pref) +"bronze_seg.nii.gz")))
        em_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(im_fn, str(im_pref) + str(app_json['output_postfix']))))
        em_seg[em_seg != 2.] = 0.
        em_seg[em_seg == 2.] = 1.
        patches = extract_random_patches(image, conf, feat, seg, em_seg, n_examples=100)
        patch_count = patch_count + 100
        for i, im in enumerate(app_json['input_postfix']):
            images[i] = np.concatenate((images[i], patches[0][:, :, :, :, i]), axis=0) \
                if (len(images[i]) != 0) else patches[0][:, :, :, :, i]

        confidences = np.concatenate((confidences, patches[1]), axis=0) \
            if (len(confidences) != 0) else patches[1]
        features = np.concatenate((features, patches[2]), axis=0) \
            if (len(features) != 0) else patches[2]
        segs = np.concatenate((segs, patches[3]), axis=0) \
            if (len(segs) != 0) else patches[3]
        em_segs = np.concatenate((em_segs, patches[4]), axis=0) \
            if (len(em_segs) != 0) else patches[4]

    # We now have our matching patches (raw image, class confidences, feature maps)
    # For each patch we flatten the confidence scores into a single value for that patch (higher = more confident)
    # This is done by summing the higher confidence score for each pixel in the patch
    confidence_vals = np.max(confidences, axis=-1)
    confidence_vals = np.sum(confidence_vals, axis=-1)
    confidence_vals = np.sum(confidence_vals, axis=-1)
    confidence_vals = confidence_vals.reshape(patch_count)
    print('conf shape:', confidence_vals.shape)

    sorted_indices = np.argsort(confidence_vals, axis=-1)
    top_conf_indices = sorted_indices[:100]  # swap sign for alternate ends of list

    top_conf_images = [[],[]]
    for i, im in enumerate(app_json['input_postfix']):
        top_conf_images[i] = [images[i][j] for j in top_conf_indices]
    top_conf_feat = [features[i] for i in top_conf_indices]
    # print(top_conf_feat.shape)
    top_conf_segs = [segs[i] for i in top_conf_indices]
    top_conf_em_segs = [em_segs[i] for i in top_conf_indices]
    # We have our most uncertain images, now use the features to select the most representative
    # Using Suggestive Annotation algorithm

    big_k = 8  # 8
    small_k = 4  # 4
    num_iterations = 4
    S_u = list(top_conf_feat)
    S_u = [np.array(x).flatten() for x in S_u]
    S_u_idx = range(len(S_u))
    S_a = []  # These are actual annotated

    for i in range(num_iterations):
        #     random_Sc_idx = random.sample(range(0,len(S_u_idx)), big_k)    # original code introduces duplicates
        random_Sc_idx = random.sample(S_u_idx, big_k)
        S_c = np.array([np.array(x).flatten() for x in top_conf_feat])
        S_c = S_c[random_Sc_idx, :]
        S_c = list(S_c)

        S_a_indices = max_rep(S_u, S_c, random_Sc_idx, small_k)
        S_a.extend(S_a_indices)
        # removes indices for next run
        S_u_idx = [x for x in range(len(S_u)) if x not in S_a]


    # Save the patches, associated segs, confs and feats
    save_dir = os.path.join(os.path.dirname(__file__), 'data', 'active_patches')
    patch_data = []
    for i, index in enumerate(S_a):
        seg = top_conf_segs[index]
        em_seg = top_conf_em_segs[index]
        sitk.WriteImage(sitk.GetImageFromArray(seg), os.path.join(save_dir, str(i) + '_seg.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(em_seg), os.path.join(save_dir, str(i) + '_emseg.nii.gz'))
        for j, im in enumerate(app_json['input_postfix']):
            patch = top_conf_images[j][index]
            sitk.WriteImage(sitk.GetImageFromArray(patch), os.path.join(save_dir, str(i) + '_' + str(im)))
        patch_data_row = [i, save_dir]
        patch_data.append(patch_data_row)

    # Update model status now that patches are available for annotation
    app_json['model_status'] = 3
    write_app_config(app_json)

    df = pd.DataFrame(patch_data, columns=["patch_id", "path"])
    df.to_csv(os.path.join(os.path.dirname(__file__), 'data', "patch_data.csv"), index=False)
    return


def small_f(S_a, I_x):
    # return the sim of the image in S_a with highest sim with I_x
    max_sim = 0
    max_sim_idx = 0
    for i, I_sa in enumerate(S_a):
        sim = cos_sim(I_sa, I_x)
        if sim > max_sim:
            max_sim = sim
            max_sim_idx = i

    return max_sim


def big_f(S_a, S_u):
    # Sum small_f for all images in S_u f_small(S_a, I_from_S_u)
    current_sum = 0
    for I_su in S_u:
        current_sum += small_f(S_a, I_su)

    return current_sum


def cos_sim(I_i, I_j):
    return np.dot(I_i, I_j.T) / I_i.shape[0] ** 2


def max_rep(S_u, S_c, Sc_idx, small_k):
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


def extract_random_patches(image_list, conf_list, feat_list, seg_list, em_seg_list,
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
        image_list = [image_list]
        conf_list = [conf_list]
        feat_list = [feat_list]
        seg_list = [seg_list]
        em_seg_list = [em_seg_list]
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
    examples_s = [[]] * len(seg_list)
    examples_e = [[]] * len(em_seg_list)
    for i in range(n_examples):
        slicer = [slice(rnd_loc[dim][i], rnd_loc[dim][i] + example_size[dim])
                  for dim in range(rank)]

        for j in range(len(image_list)):
            ex_image = image_list[j][slicer][np.newaxis]
            ex_conf = conf_list[j][slicer][np.newaxis]
            ex_feat = feat_list[j][slicer][np.newaxis]
            ex_seg = seg_list[j][slicer][np.newaxis]
            ex_emseg = em_seg_list[j][slicer][np.newaxis]
            # Concatenate and return the examples
            examples[j] = np.concatenate((examples[j], ex_image), axis=0) \
                if (len(examples[j]) != 0) else ex_image
            examples_f[j] = np.concatenate((examples_f[j], ex_feat), axis=0) \
                if (len(examples_f[j]) != 0) else ex_feat
            examples_c[j] = np.concatenate((examples_c[j], ex_conf), axis=0) \
                if (len(examples_c[j]) != 0) else ex_conf
            examples_s[j] = np.concatenate((examples_s[j], ex_seg), axis=0) \
                if (len(examples_s[j]) != 0) else ex_seg
            examples_e[j] = np.concatenate((examples_e[j], ex_emseg), axis=0) \
                if (len(examples_e[j]) != 0) else ex_emseg

    if was_singular:
        return [examples[0], examples_c[0], examples_f[0], examples_s[0], examples_e[0]]
    return [examples, examples_c, examples_f, examples_s, examples_e]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dhcp brain segmentation deploy')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p',
                        default=os.path.join(os.path.dirname(__file__), 'model'))

    parser.add_argument('--csv',
                        default=os.path.join(os.path.dirname(__file__), 'data', 'subject_data.csv'))

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    #select_patches()
    select_patch_batch(args, get_config_for_app())
