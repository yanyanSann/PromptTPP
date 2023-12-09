""" Base model with common functionality  """

import torch
from torch import nn

from neural_tpp.model.torch_model.torch_thinning import EventSampler
from neural_tpp.utils import set_device


class TorchBaseModel(nn.Module):
    def __init__(self, model_config):
        super(TorchBaseModel, self).__init__()
        self.device = set_device(model_config['gpu'])
        model_config.update({'device': self.device})
        self.loss_integral_num_sample_per_step = model_config['loss_integral_num_sample_per_step']
        self.hidden_size = model_config['hidden_size']
        self.num_event_types_no_pad = model_config['num_event_types_no_pad']
        self.num_event_types_pad = model_config['num_event_types_pad'] 
        self.event_pad_index = model_config['event_pad_index']
        self.eps = torch.finfo(torch.float32).eps

        self.layer_type_emb = nn.Embedding(self.num_event_types_pad,
                                           self.hidden_size,
                                           padding_idx=self.event_pad_index)

        self.gen_config = model_config.get('thinning_params', None)
        self.event_sampler = None
        if self.gen_config:
            self.event_sampler = EventSampler(num_sample=self.gen_config['num_sample'],
                                              num_exp=self.gen_config['num_exp'],
                                              over_sample_rate=self.gen_config['over_sample_rate'],
                                              patience_counter=self.gen_config['patience_counter'],
                                              num_samples_boundary=self.gen_config['num_samples_boundary'],
                                              dtime_max=self.gen_config['dtime_max'],
                                              device=self.device)

    def model_to_device(self):
        self.to(device=self.device)

    @staticmethod
    def generate_model_from_config(model_config):
        model_name = model_config['name']

        for subclass in TorchBaseModel.__subclasses__():
            if subclass.__name__ == model_name:
                return subclass(model_config)

        raise RuntimeError('No model named ' + model_name)

    @staticmethod
    def get_logits_at_last_step(logits, batch_non_pad_mask, sample_len=None):

        seq_len = batch_non_pad_mask.sum(dim=1)
        select_index = seq_len - 1 if sample_len is None else seq_len - 1 - sample_len
        # [batch_size, hidden_dim]
        select_index = select_index.unsqueeze(1).repeat(1, logits.size(-1))
        # [batch_size, 1, hidden_dim]
        select_index = select_index.unsqueeze(1)
        # [batch_size, hidden_dim]
        last_logits = torch.gather(logits, dim=1, index=select_index).squeeze(1)
        return last_logits

    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask,
                              lambda_type_mask):
        event_lambdas = torch.sum(lambda_at_event * lambda_type_mask, dim=-1) + self.eps

        event_lambdas = event_lambdas.masked_fill_(~seq_mask, 1.0)

        # [batch_size, n_loss_sample, num_times)
        event_ll = torch.log(event_lambdas)

        lambdas_total_samples = lambdas_loss_samples.sum(dim=-1)

        non_event_ll = lambdas_total_samples.mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]

        return event_ll, non_event_ll, num_events

    def make_dtime_loss_samples(self, time_delta_seq):
        dtimes_ratio_sampled = torch.linspace(start=0.0,
                                              end=1.0,
                                              steps=self.loss_integral_num_sample_per_step,
                                              device=self.device)[None, None, :]

        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes

    def compute_states_at_sample_times(self, **kwargs):

        raise NotImplementedError('This need to implemented in inherited class ! ')

    def prediction_event_one_step(self, batch):
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _, type_mask = batch

        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        dtime_boundary = time_delta_seq + self.event_sampler.dtime_max

        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(time_seq,
                                                                              time_delta_seq,
                                                                              event_seq,
                                                                              dtime_boundary,
                                                                              self.compute_intensities_at_sample_times)

        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)

        intensities_at_times = self.compute_intensities_at_sample_times(time_seq,
                                                                        time_delta_seq,
                                                                        event_seq,
                                                                        dtimes_pred[:, :, None],
                                                                        max_steps=event_seq.size()[1])

        intensities_at_times = intensities_at_times.squeeze(dim=-2)

        types_pred = torch.argmax(intensities_at_times, dim=-1)

        return dtimes_pred, types_pred

    def prediction_event_multi_step(self, batch, num_steps):
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _, type_mask = batch

        dtime_boundary = time_delta_seq + self.event_sampler.dtime_max

        for i in range(num_steps):
            accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(time_seq,
                                                                                  time_delta_seq,
                                                                                  event_seq,
                                                                                  dtime_boundary,
                                                                                  self.compute_intensities_at_sample_times)

            dtimes_pred = torch.sum(accepted_dtimes[:, -1, :] * weights[:, -1, :], dim=-1)

            intensities_at_times = self.compute_intensities_at_sample_times(time_seq,
                                                                            time_delta_seq,
                                                                            event_seq,
                                                                            dtimes_pred[:, :, None],
                                                                            max_steps=event_seq.size()[1])

            intensities_at_times = intensities_at_times.squeeze(dim=-2)

            types_pred = torch.argmax(intensities_at_times, dim=-1)

            time_pred = time_seq[:, -1:] + dtimes_pred

            time_seq = torch.concat([time_seq, time_pred], dim=-1)
            time_delta_seq = torch.concat([time_delta_seq, dtimes_pred], dim=-1)
            event_seq = torch.concat([event_seq, types_pred], dim=-1)

        return time_delta_seq[:, -num_steps:], event_seq[:, -num_steps:]
