from __future__ import division
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk
from importlib import import_module
import json
from tensorflow.contrib import predictor
from dltk.core import metrics as metrics
from dltk.utils import sliding_window_segmentation_inference


def test_app_func(model_name='model'):
    # Set up argument parser
    parser = argparse.ArgumentParser(description='DALTK ALUI')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p',
                        default=os.path.join(os.path.dirname(__file__), model_name))

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
    predict(args)


def predict(args):
    file_names = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()

    # We trained on the first 4 subjects, so we predict on the rest
    test_filenames = []
    for i, row in enumerate(file_names):
        if row[5] == '1':
            test_filenames.append(row)
    print('testing on : ', len(test_filenames), ' entries')
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

    mod = import_module('readers.slice_reader')
    # mod = import_module(module_name)
    read_fn = vars(mod)['read_fn']
    reader_params = {'extract_examples': False}

    results = []
    print("Preparing to predict")
    # Iterate through the files, predict on the full volumes and compute a Dice
    # coefficient
    for output in read_fn(file_references=test_filenames,
                          mode=tf.estimator.ModeKeys.EVAL,
                          params=reader_params):
        print("Predicting on an entry")
        t0 = time.time()
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

        # Calculate the prediction from the probabilities
        pred = np.argmax(pred, -1)

        # Calculate the Dice coefficient
        dsc = metrics.dice(pred, lbl, num_classes)[1:].mean()

        # Print outputs
        print('Id={}; Dice={:0.4f}; time={:0.2} secs;'.format(
            output['subject_id'], dsc, time.time() - t0))
        res_row = [output['subject_id'], dsc, time.time() - t0]
        results.append(res_row)

    df = pd.DataFrame(results, columns=["ID", "Dice", "Time"])
    df.to_csv(os.path.join(args.model_path, 'test_results.csv'), index=False)


if __name__ == '__main__':
    print('test from main')
    test_app_func()
