# Script run when first creating application
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import threading
import argparse
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import json
from importlib import import_module

from dltk.core.metrics import dice
from dltk.networks.segmentation.unet import residual_unet_3d
from dltk.io.abstract_reader import Reader

EVAL_EVERY_N_STEPS = 100
EVAL_STEPS = 1
SHUFFLE_CACHE_SIZE = 64


# Entry point
def retrain_model_func():
    print('retrain!')
    app_json = get_config_for_app()

    model_status = int(app_json['model_status'])
    new_model_status = -1
    if model_status == 0:
        new_model_status = 0
    elif model_status == 1:
        new_model_status = 1
    elif model_status == 2:
        new_model_status = 2
    elif model_status == 3:
        # Patches available for annotation
        new_model_status = 3
    elif model_status == 4:
        # Annotated Patches available for retraining
        # TODO: trigger retraining/fine-tuning
        new_model_status = 1

    return new_model_status


def tune_model():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Contribution: dHCP GM AL framework training")
    parser.add_argument('--run_validation', default=True)
    parser.add_argument('--restart', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    next_model_iteration_num = get_config_for_app()['model_iteration'] + 1

    parser.add_argument('--model_path', '-p',
                        default=os.path.join(os.path.dirname(__file__), 'model_' + str(next_model_iteration_num)))

    parser.add_argument('--train_csv',
                        default=os.path.join(os.path.dirname(__file__), 'data', 'subject_data'))

    args = parser.parse_args()

    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Handle restarting and resuming training
    if args.restart:
        print('Restarting training from scratch.')
        os.system('rm -rf {}'.format(args.model_path))

    if not os.path.isdir(args.model_path):
        os.system('mkdir -p {}'.format(args.model_path))
    else:
        print('Resuming training on model_path {}'.format(args.model_path))

    # Call training
    train(args)


def get_config_for_app():
    app_fn = os.path.join(os.path.dirname(__file__), 'app_config.json')
    with open(app_fn) as json_data:
        app_json = json.load(json_data)
    return app_json


def write_app_config(app_json):
    app_fn = os.path.join(os.path.dirname(__file__), 'app_config.json')
    with open(app_fn, 'w') as outfile:
        json.dump(app_json, outfile, indent=4)