import argparse
from neural_tpp.model_runner import ModelRunner
from neural_tpp.utils import load_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='example_config',
                        help='Configuration folder dir to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='PromptAttNHP_train',
                        help='Experiment id in the config file.')

    args = parser.parse_args()

    config = load_config(args.config_dir, args.experiment_id)

    model_runner = ModelRunner(config)

    model_runner.train()


if __name__ == '__main__':
    main()
