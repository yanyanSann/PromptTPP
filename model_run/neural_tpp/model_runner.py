import os
from collections import OrderedDict
from datetime import datetime

import numpy as np

from neural_tpp.model import TorchBaseModel
from neural_tpp.preprocess import TorchTPPDataset, create_torch_dataloader
from neural_tpp.torch_wrapper import TorchModelWrapper
from neural_tpp.utils import TPPMetrics, has_key, set_seed, array_pad_cols
from neural_tpp.utils import create_folder, load_pickle, py_assert, \
    save_config, RunnerPhase, LogHandler


class ModelRunner:
    def __init__(self, config):
        """
        Args:
            config: dict of config
        """

        self.config = config

        # data config could possibly be None, e.g., during generation
        self.data_config = self.config.get('data', None)

        # Base config is mandatory
        self.base_config = config.get(config['model_id']).get('base')

        # Model config is mandatory
        self.model_config = config.get(config['model_id']).get('model')

        self.ensure_valid_config()

        self.use_torch = self.base_config['use_torch']

        # by default we dont use tensor board
        self.use_tfb = self.base_config['use_tfb']

        self.update_model_config = True

        self.update_config()

        self.init_log()

        self.build_runner()

        self.save_updated_config()

        if 'metrics' in self.base_config:
            self.metrics_function = TPPMetrics.get_metrics_callback_from_names(self.base_config['metrics'])

    def ensure_valid_config(self):
        """Do some sanity check about the config, to avoid conflicts in settings
        """
        # in training mode, one must set up train_config
        is_training = self.base_config['is_training']

        if is_training:
            py_assert(has_key(self.base_config, ['batch_size', 'max_epoch', 'optimizer', 'learning_rate', 'valid_freq',
                                                 'use_tfb', 'metrics']), ValueError,
                      'Missing train configs in training mode (is_training=True)')

        else:
            # during testing we dont do shuffle by default
            if not hasattr(self.base_config, 'shuffle'):
                self.base_config['shuffle'] = False

            # during testing we dont apply tfb by default
            if not hasattr(self.base_config, 'use_tfb'):
                self.base_config['use_tfb'] = False

            # during testing we use test loader for testing by default
            if not hasattr(self.base_config, 'target_loader'):
                self.base_config['target_loader'] = 'test'

        return

    def init_log(self):
        """Initialize the logger
        """

        self.log = LogHandler(self.base_config['saved_log_dir'])
        self.log.init(self.config)

        return

    def update_config(self):
        """Updated config dict
        """
        time = datetime.now()
        timestamp = datetime.strftime(time, '%Y%m%d-%H:%M:%S')
        model_id = self.config['model_id']
        dataset_id = self.model_config['dataset_id']
        model_folder_name = model_id + '_' + dataset_id + '_' + timestamp

        self.log_folder = create_folder(self.base_config['base_dir'], model_folder_name)
        self.model_folder = create_folder(self.log_folder, 'models')

        self.base_config['log_folder'] = self.log_folder
        self.base_config['saved_model_dir'] = os.path.join(self.model_folder, 'saved_model')
        self.base_config['saved_log_dir'] = os.path.join(self.log_folder, 'log')
        self.base_config['output_config_dir'] = os.path.join(self.log_folder, f'{model_id}_output.yaml')

        if self.use_tfb:
            self.base_config['tfb_train_dir'] = create_folder(self.log_folder, 'tfb_train')
            self.base_config['tfb_valid_dir'] = create_folder(self.log_folder, 'tfb_valid')

        if not self.use_torch:
            self.model_config['is_training'] = self.base_config['is_training']

        return

    def save_updated_config(self):
        """Update the config dict that is to be saved
        """

        if has_key(self.base_config, 'metrics') and not has_key(self.model_config, 'thinning_params'):
            self.log.warning(
                'The metrics has no effect as thinning params has not been filled: the evaluation needs thinning '
                'params to set up an event sampler')

        run = 'Train' if self.base_config['is_training'] else 'Evaluate'
        model_name = self.model_config['name']
        tf_torch = 'PyTorch' if self.base_config['use_torch'] else 'Tensorflow'
        device = 'GPU' if self.model_config['gpu'] >= 0 else 'CPU'
        critical_msg = '{run} model {model_name} with {device} with {tf_torch} backend'.format(run=run,
                                                                                               model_name=model_name,
                                                                                               device=device,
                                                                                               tf_torch=tf_torch)

        self.log.critical(critical_msg)

        save_config(self.base_config['output_config_dir'], self.config)
        return

    def build_runner(self):
        """Build up dataloader, model and model wrapper
        """
        if self.use_torch:
            # set up random seed
            set_seed(self.model_config['seed'])

            [self.train_loader, self.dev_loader, self.test_loader] \
                = self.get_dataloader(TorchTPPDataset, create_torch_dataloader)
            self.model = TorchBaseModel.generate_model_from_config(model_config=self.model_config)

            self.model_wrapper = TorchModelWrapper(self.model, self.base_config, self.model_config)
        else:
            print('The model currently only supports Torch. ')

        return

    def get_dataloader(self, dataset_cls, dataloader_fn=None, num_event_types=None):
        """Assume that we load data from pickle files with GaTech format.
        One can modify this function to accommodate other customized format.

        Args:
            dataset_cls: a Tf or Torch dataset class object
            dataloader_fn: a mapper from dataset object to data loader
            num_event_types: default None


        Returns:
            train, dev and test dataloader

        """

        loaders = []
        splits = ['train', 'dev', 'test']
        for _split in splits:
            with open(self.data_config.get('{}_data'.format(_split)), "rb") as f_in:
                data = load_pickle(f_in)

                if num_event_types is None:
                    num_event_types = data["dim_process"]
                else:
                    py_assert(data["dim_process"] == num_event_types,
                              ValueError,
                              "inconsistent dim_process in different splits?")

                dataset = dataset_cls(dict(event_num=num_event_types,
                                           source_data=data[_split]),
                                      batch_size=self.base_config['batch_size'])

                loaders.append(dataloader_fn(dataset,
                                             batch_size=self.base_config['batch_size'],
                                             shuffle=self.base_config['shuffle']))

                if self.update_model_config:
                    self.model_config['num_event_types_pad'] = dataset.num_event_types_pad
                    self.model_config['num_event_types_no_pad'] = dataset.num_event_types_no_pad
                    self.model_config['event_pad_index'] = dataset.event_pad_index
                    self.update_model_config = False

        return loaders

    def train(self):
        """train the model
        """

        for i in range(self.base_config['max_epoch']):
            train_metrics = self.run_one_epoch(self.train_loader, RunnerPhase.TRAIN)

            # lr_scheduler
            # self.model_wrapper.scheduler.step()


            message = f"[ Epoch {i} (train) ]: train " + TPPMetrics.metrics_dict_to_str(train_metrics)
            self.log.info(message)

            self.model_wrapper.write_summary(i, train_metrics, RunnerPhase.TRAIN)

            # evaluate model
            if i % self.base_config['valid_freq'] == 0:
                valid_metrics = self.run_one_epoch(self.dev_loader, RunnerPhase.VALIDATE)

                self.model_wrapper.write_summary(i, valid_metrics, RunnerPhase.VALIDATE)

                message = f"[ Epoch {i} (valid) ]:  valid " + TPPMetrics.metrics_dict_to_str(valid_metrics)
                self.log.info(message)

                updated = self.log.update_best("loglike", valid_metrics['loglike'], i)

                message = "current best loglike is {:.4f} (updated at epoch-{})".format(
                    self.log.current_best['loglike'], self.log.episode_best)
                self.log.critical(message)

                if updated:
                    message += f", best updated at this epoch"
                    self.model_wrapper.save(self.base_config['saved_model_dir'])

                test_metrics = self.run_one_epoch(self.test_loader, RunnerPhase.VALIDATE)
                

                message = f"[ Epoch {i} (test) ]: test " + TPPMetrics.metrics_dict_to_str(test_metrics)
                self.log.info(message)

        self.model_wrapper.close_summary()

        return

    def eval(self):
        """Perform the one step prediction given the ground truth sequence
        """

        data_loader = self.test_loader if self.base_config['target_loader'] == 'test' else self.dev_loader

        test_metrics = self.run_one_epoch(data_loader, RunnerPhase.VALIDATE)

        self.model_wrapper.write_summary(0, test_metrics, RunnerPhase.VALIDATE)

        self.model_wrapper.close_summary()

        message = f"Evaluation result: " + TPPMetrics.metrics_dict_to_str(test_metrics)
        self.log.critical(message)

        return

    def run_one_epoch(self, data_loader, phase):
        """Run one complete epoch

        Args:
            data_loader: data loader object defined in model runner
            phase: enum, [train, dev, test]

        Returns:
            a dict of metrics

        """
        total_loss = 0
        total_num_event = 0
        epoch_label = []
        epoch_pred = []
        epoch_mask = []
        pad_index = self.data_config['event_pad_index']
        for batch in data_loader:
            batch_loss, batch_num_event, batch_pred, batch_label, batch_mask = self.model_wrapper.run_batch(batch,
                                                                                                            phase=phase)

            total_loss += batch_loss
            total_num_event += batch_num_event
            epoch_pred.append(batch_pred)
            epoch_label.append(batch_label)
            epoch_mask.append(batch_mask)

            # classify batch_output to list
        pred_exists, label_exists = False, False
        if epoch_pred[0][0] is not None:
            epoch_pred = self.concat_element(epoch_pred, pad_index)
            pred_exists = True
        if epoch_label[0][0] is not None:
            epoch_label = self.concat_element(epoch_label, pad_index)
            label_exists = True
            epoch_mask = self.concat_element(epoch_mask, False)[0]  # retrieve the first element of concat array
            epoch_mask = epoch_mask.astype(bool)

        avg_loss = total_loss / total_num_event

        metrics_dict = OrderedDict()
        metrics_dict.update({'loglike': -avg_loss, 'num_events': total_num_event})

        if pred_exists and label_exists:
            metrics_dict.update(self.metrics_function(epoch_pred, epoch_label, seq_mask=epoch_mask))

        return metrics_dict

    @staticmethod
    def concat_element(arrs, pad_index):
        """ Concat element from each batch output  """

        n_lens = len(arrs)
        n_elements = len(arrs[0])

        # found out the max seq len (num cols) in output arrays
        max_len = max([x[0].shape[1] for x in arrs])

        concated_outputs = []
        for j in range(n_elements):
            a_output = []
            for i in range(n_lens):
                arrs_ = array_pad_cols(arrs[i][j], max_num_cols=max_len, pad_index=pad_index)
                a_output.append(arrs_)

            concated_outputs.append(np.concatenate(a_output, axis=0))

        # n_elements * [ [n_lens, dim_of_element] ]
        return concated_outputs
