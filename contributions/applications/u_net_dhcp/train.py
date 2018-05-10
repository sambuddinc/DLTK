from reader import read_fn

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

    return


def train(args):
    print("training")
    return


if __name__ == '__main__':
    print("main")
    args = {}
    train(args)
