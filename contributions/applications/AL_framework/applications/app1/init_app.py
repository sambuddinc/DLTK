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

def init_app_func():
    # Start app initialisation
    print('init app!')

    # Load app config -> check model status before initialising
    app_json = get_config_for_app()

    print(app_json)
    model_status = int(app_json['model_status'])
    new_model_status = -1
    if model_status == 0:
        # no model exists yet: init the model and begin training in new thread.
        # load variables from app_config for model start
        # set model status to 1 (training) so we dont initialise again
        new_model_status = 1
        app_json['model_status'] = new_model_status
        write_app_config(app_json)
        # set up thread to start training
        train_app_thread = threading.Thread(target=train_initial_model, args={})
        train_app_thread.start()

    elif model_status == 1:
        # model is under training -> do nothing
        new_model_status = 1

    elif model_status == 2:
        # model is ready -> do nothing
        new_model_status = 2

    return new_model_status


def train_initial_model():
    # Function to train model for current app
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Contribution: dHCP GM AL framework training")
    parser.add_argument('--run_validation', default=True)
    parser.add_argument('--restart', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p',
                        default=os.path.join(os.path.dirname(__file__), 'model'))

    parser.add_argument('--train_csv',
                        default=os.path.join(os.path.dirname(__file__), 'data', 'subject_data.csv'))

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


def train(args):
    np.random.seed(42)
    tf.set_random_seed(42)

    print('Setting up...')

    # Parse csv files for file names
    all_filenames = pd.read_csv(
        args.train_csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()

    app_json = get_config_for_app()

    # train_val_test_split = app_json['train_val_test_split']
    #
    # num_train = train_val_test_split[0]
    # num_val = train_val_test_split[1]
    #
    # train_filenames = all_filenames[0:num_train]
    # val_filenames = all_filenames[num_train:num_val]

    train_filenames = []
    val_filenames = []

    for row in all_filenames:
        if row[3] == '1':
            train_filenames.append(row)
        if row[4] == '1':
            val_filenames.append(row)

    # Set up a data reader to handle the file i/o.
    reader_params = {'n_examples': 16,
                     'example_size': [1, 64, 64],
                     'extract_examples': True}
    num_channels = app_json['num_channels']
    reader_example_shapes = {'features': {'x': reader_params['example_size'] + [num_channels, ]},
                             'labels': {'y': reader_params['example_size']}}

    #module_name = 'contributions.applications.AL_framework.applications.app' + str(app_json['id']) + '.readers.'
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

    mod = import_module(module_name)
    read_fn = vars(mod)['read_fn']

    reader = Reader(read_fn,
                    {'features': {'x': tf.float32},
                     'labels': {'y': tf.int32}})

    # Get input functions and queue initialisation hooks for training and
    # validation data
    batch_size = app_json['batch_size']
    train_input_fn, train_qinit_hook = reader.get_inputs(
        file_references=train_filenames,
        mode=tf.estimator.ModeKeys.TRAIN,
        example_shapes=reader_example_shapes,
        batch_size=batch_size,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE,
        params=reader_params)

    val_input_fn, val_qinit_hook = reader.get_inputs(
        file_references=val_filenames,
        mode=tf.estimator.ModeKeys.EVAL,
        example_shapes=reader_example_shapes,
        batch_size=batch_size,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE,
        params=reader_params)

    # Instantiate the neural network estimator
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_path,
        params={"learning_rate": 0.001},
        config=tf.estimator.RunConfig())

    # Hooks for validation summaries
    val_summary_hook = tf.contrib.training.SummaryAtEndHook(
        os.path.join(args.model_path, 'eval'))
    step_cnt_hook = tf.train.StepCounterHook(
        every_n_steps=EVAL_EVERY_N_STEPS,
        output_dir=args.model_path)

    print('Starting training...')
    max_steps = app_json['max_steps']
    try:
        for _ in range(max_steps // EVAL_EVERY_N_STEPS):
            nn.train(
                input_fn=train_input_fn,
                hooks=[train_qinit_hook, step_cnt_hook],
                steps=EVAL_EVERY_N_STEPS)

            if args.run_validation:
                results_val = nn.evaluate(
                    input_fn=val_input_fn,
                    hooks=[val_qinit_hook, val_summary_hook],
                    steps=EVAL_STEPS)
                print('Step = {}; val loss = {:.5f};'.format(
                    results_val['global_step'], results_val['loss']))

    except KeyboardInterrupt:
        pass

    print('Stopping now.')
    export_dir = nn.export_savedmodel(
        export_dir_base=args.model_path,
        serving_input_receiver_fn=reader.serving_input_receiver_fn(reader_example_shapes))
    print('Model saved to {}.'.format(export_dir))
    app_json['model_status'] = 2
    write_app_config(app_json)
    print('Updated model status in model config')


def model_fn(features, labels, mode, params):
    """Model function to construct a tf.estimator.EstimatorSpec. It creates a
            network given input features (e.g. from a dltk.io.abstract_reader) and
            training targets (labels). Further, loss, optimiser, evaluation ops and
            custom tensorboard summary ops can be added. For additional information,
            please refer to https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#model_fn.

        Args:
            features (tf.Tensor): Tensor of input features to train from. Required
                rank and dimensions are determined by the subsequent ops
                (i.e. the network).
            labels (tf.Tensor): Tensor of training targets or labels. Required rank
                and dimensions are determined by the network output.
            mode (str): One of the tf.estimator.ModeKeys: TRAIN, EVAL or PREDICT
            params (dict, optional): A dictionary to parameterise the model_fn
                (e.g. learning_rate)

        Returns:
            tf.estimator.EstimatorSpec: A custom EstimatorSpec for this experiment
        """
    print("Setting up U-Net")
    app_json = get_config_for_app()
    # 1. create a model and its outputs
    net_output_ops = residual_unet_3d(
        inputs=features['x'],
        num_classes=app_json['num_classes'],
        num_res_units=2,
        filters=(16, 32, 64, 128),
        strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
        mode=mode,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

    # 1.1 Generate predictions only (for `ModeKeys.PREDICT`)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=net_output_ops,
            export_outputs={'out': tf.estimator.export.PredictOutput(net_output_ops)})

    # 2. set up a loss function
    # print(labels['y'])
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=net_output_ops['logits'],
        labels=labels['y'])
    loss = tf.reduce_mean(ce)

    # 3. define a training op and ops for updating moving averages
    # (i.e. for batch normalisation)
    global_step = tf.train.get_global_step()
    optimiser = tf.train.MomentumOptimizer(
        learning_rate=params["learning_rate"],
        momentum=0.9)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimiser.minimize(loss, global_step=global_step)

    # 4.1 (optional) create custom image summaries for tensorboard
    my_image_summaries = {}
    for n, input_type in enumerate(app_json['input_desc']):
        my_image_summaries[input_type] = features['x'][0, 0, :, :, n]
    my_image_summaries['labels'] = tf.cast(labels['y'], tf.float32)[0, 0, :, :]
    my_image_summaries['predictions'] = tf.cast(net_output_ops['y_'], tf.float32)[0, 0, :, :]

    expected_output_size = [1, 64, 64, 1]  # [B, W, H, C]
    [tf.summary.image(name, tf.reshape(image, expected_output_size))
     for name, image in my_image_summaries.items()]

    # 4.2 (optional) create custom metric summaries for tensorboard
    dice_tensor = tf.py_func(dice, [net_output_ops['y_'],
                                    labels['y'],
                                    tf.constant(app_json['num_classes'])], tf.float32)
    [tf.summary.scalar('dsc_l{}'.format(i), dice_tensor[i])
     for i in range(app_json['num_classes'])]

    # 5. Return EstimatorSpec object
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=net_output_ops,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=None)


if __name__ == '__main__':
    print('init from main')
    train_initial_model()

