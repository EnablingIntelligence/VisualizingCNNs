from train import test
from utils import parse_args

if __name__ == "__main__":
    config = parse_args()

    if config.train:
        print("Start training the model")
        # TODO #11 pylint: disable=fixme
    else:
        print("Start testing the model")
        test(config)
