""" Initialize a Pytorch model wrapper that feed into Model Runner   """

import torch
from torch.utils.tensorboard import SummaryWriter

from neural_tpp.utils import RunnerPhase, set_optimizer, set_device
from neural_tpp.utils.lr_scheduler import WarmupMultiStepLR


class TorchModelWrapper:
    def __init__(self, model, base_config, model_config):
        self.model = model
        self.base_config = base_config
        self.model_config = model_config
        self.device = set_device(model_config['gpu'])
        if self.base_config['is_training']:
            # set up optimizer
            optimizer = self.base_config['optimizer']
            self.learning_rate = self.base_config['learning_rate']
            self.opt = set_optimizer(optimizer, self.model.parameters(), self.learning_rate)

        # set up tensorboard
        self.use_tfb = self.base_config['use_tfb']
        self.train_summary_writer, self.valid_summary_writer = None, None
        if self.use_tfb:
            self.train_summary_writer = SummaryWriter(log_dir=self.base_config['tfb_train_dir'])
            self.valid_summary_writer = SummaryWriter(log_dir=self.base_config['tfb_valid_dir'])
        
        self.refresh_flag = True

        if self.base_config['pretrained_model_dir'] != 'None':
            print('Loading Pretrained Model')
            self.restore(self.base_config['pretrained_model_dir'])

        self.scheduler = WarmupMultiStepLR(self.opt,
            [80,90],
            gamma=0.01,
            warmup_epochs=3,
        )



    def restore(self, ckpt_dir):
        """

        Args:
            ckpt_dir: dir to load the model

        Returns:
            load the model

        """

        ## self.model.load_state_dict(torch.load(ckpt_dir), strict=False)

        if self.base_config['pretrain_type'] == 'Prompt_only':
            model_dict=self.model.heads.state_dict()
            pretrained_dict = {k: v for k, v in torch.load(ckpt_dir).items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
        else:
            self.model.load_state_dict(torch.load(ckpt_dir), strict=False)

    def save(self, ckpt_dir):
        """

        Args:
            ckpt_dir: dir to save the model

        Returns:
            save the model

        """
        torch.save(self.model.state_dict(), ckpt_dir)

    def write_summary(self, epoch, kv_pairs, phase):
        """

        Args:
            epoch: epoch index in the training
            kv_pairs: metrics dict
            phase: RunnerPhase

        Returns:
            write the kv_paris into the tensorboard

        """
        if self.use_tfb:
            summary_writer = None
            if phase == RunnerPhase.TRAIN:
                summary_writer = self.train_summary_writer
            elif phase == RunnerPhase.VALIDATE:
                summary_writer = self.valid_summary_writer
            elif phase == RunnerPhase.PREDICT:
                pass

            if summary_writer is not None:
                for k, v in kv_pairs.items():
                    if k != 'num_events':
                        summary_writer.add_scalar(k, v, epoch)

                summary_writer.flush()
        return

    def close_summary(self):
        if self.train_summary_writer is not None:
            self.train_summary_writer.close()

        if self.valid_summary_writer is not None:
            self.valid_summary_writer.close()
        return

    def run_batch(self, batch, phase):
        """  Run one batch """

        batch = [x.to(self.device) for x in batch]
        if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
            # set mode to train
            is_training = (phase == RunnerPhase.TRAIN)
            self.model.train(is_training)

            # run model
            with torch.set_grad_enabled(is_training):
                loss, num_event = self.model.loglike_loss(batch)

            # Assume we dont do prediction on train set
            pred_dtime, pred_type = None, None
            label_dtime, label_type = batch[1][:, 1:].cpu().numpy(), batch[2][:, 1:].cpu().numpy()
            mask = batch[3][:, 1:].cpu().numpy()
            # update grad
            if is_training:
                self.opt.zero_grad()
                if self.refresh_flag:
                    self.model.warm_prompt_layer()
                else:
                    self.model.freeze_prompt_layer()
                self.refresh_flag = not self.refresh_flag
                loss.backward()
                self.opt.step()
            else:
                if self.model.event_sampler:
                    pred_dtime, pred_type = self.model.prediction_event_one_step(batch=batch)
                    pred_dtime = pred_dtime.detach().cpu().numpy()
                    pred_type = pred_type.detach().cpu().numpy()

            return loss.item(), num_event, (pred_dtime, pred_type), (label_dtime, label_type), (mask,)
        else:
            pred_time, pred_type = self.model.predict(batch)
            return pred_time.numpy(), pred_type.numpy()
