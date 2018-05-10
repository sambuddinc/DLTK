

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

        return img, lbl

    for f in file_references:
        print(f)

    print('read_fn booom')

    return

