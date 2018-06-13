from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk

from tensorflow.contrib import predictor

from dltk.core import metrics as metrics

from dltk.utils import sliding_window_segmentation_inference

from reader import read_fn

READER_PARAMS = {'extract_examples': False}
N_VALIDATION_SUBJECTS = 89

def predict(args):
    file_names = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()

    # We trained on the first 4 subjects, so we predict on the rest
    file_names = file_names[-N_VALIDATION_SUBJECTS:]

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

    results = []
    print("Preparing to predict")
    # Iterate through the files, predict on the full volumes and compute a Dice
    # coefficient
    for output in read_fn(file_references=file_names,
                          mode=tf.estimator.ModeKeys.EVAL,
                          params=READER_PARAMS):
        t0 = time.time()
        print("Predicting on an entry")
        # Parse the read function output and add a dummy batch dimension as
        # required
        img = np.expand_dims(output['features']['x'], axis=0)
        lbl = np.expand_dims(output['labels']['y'], axis=0)
        # Do a sliding window inference with our DLTK wrapper
        pred = sliding_window_segmentation_inference(
            session=my_predictor.session,
            ops_list=[y_prob],
            sample_dict={my_predictor._feed_tensors['x']: img},
            batch_size=16)[0]

        print("Prediction: " + str(pred.shape))

        features = sliding_window_segmentation_inference(
            session=my_predictor.session,
            ops_list=[logits],
            sample_dict={my_predictor._feed_tensors['x']: img},
            batch_size=16)[0]

        class_confidences = pred

        # Calculate the prediction from the probabilities
        pred = np.argmax(pred, -1)

        # Calculate the Dice coefficient
        dsc = metrics.dice(pred, lbl, num_classes)[1:].mean()

        # Calculate the cross entropy coeff
        #cross_ent = metrics.crossentropy(features, lbl)
        cross_ent = "error"

        # Save the file as .nii.gz using the header information from the
        # original sitk image
        output_fn = os.path.join(args.model_path, '{}_seg.nii.gz'.format(output['subject_id']))

        new_sitk = sitk.GetImageFromArray(pred[0].astype(np.int32))
        new_sitk.CopyInformation(output['sitk'])

        sitk.WriteImage(new_sitk, output_fn)
        
        # Save the feature vector file as a .nii.gz using header info from origincal sitk
        print("Features: " + str(features.shape))
        feature_sitk = sitk.GetImageFromArray(features[0])
        feature_sitk.CopyInformation(output['sitk'])
        sitk.WriteImage(feature_sitk, os.path.join(args.model_path, 'ALout', '{}_feat.nii.gz'.format(output['subject_id'])))

        # Save the confidence vector file as a .nii.gz using header info from original stack
        print("Confidences: " + str(class_confidences.shape))
        conf_sitk = sitk.GetImageFromArray(class_confidences[0])
        conf_sitk.CopyInformation(output['sitk'])
        sitk.WriteImage(conf_sitk, os.path.join(args.model_path, 'ALout', '{}_conf.nii.gz'.format(output['subject_id'])))

        # Print outputs
        print('Id={}; Dice={:0.4f}; time={:0.2} secs; output_path={};'.format(
            output['subject_id'], dsc, time.time() - t0, output_fn))
        res_row = [output['subject_id'], dsc, cross_ent, time.time() - t0, output_fn]
        results.append(res_row)

    df = pd.DataFrame(results, columns=["ID", "Dice", "Cross Entropy", "Time", "Segmentation Path"])
    df.to_csv(os.path.join(args.model_path, 'ALout', 'results_baseline_alfetest.csv'), index=False)


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='dhcp brain segmentation deploy')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p', default='/home/sb17/DLTK/contributions/applications/baseline/baseline_model')
    parser.add_argument('--csv', default='/home/sb17/DLTK/contributions/applications/baseline/experiment_baseline.csv')

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
