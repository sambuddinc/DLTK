from reader import read_fn


def predict(args):
    print(args)
    read_fn({}, None)
    return


if __name__ == '__main__':
    print('main deploy')
    args = {}
    predict(args)
