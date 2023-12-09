import argparse
from neural_tpp.model_runner import ModelRunner
from neural_tpp.utils import load_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=False, default='step_1_train_nhp.yaml',
                        help='Configuration dir to train and evaluate the model.')

    args = parser.parse_args()

    config = load_config(args.config)

    model_runner = ModelRunner(config)

    model_runner.train()


if __name__ == '__main__':
    main()
