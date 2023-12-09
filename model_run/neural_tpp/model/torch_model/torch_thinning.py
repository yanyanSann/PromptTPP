import torch
import torch.nn as nn


class EventSampler(nn.Module):
    """
    Event Sequence Sampler based on thinning algorithm

    The algorithm can be found at Algorithm 2 of The Neural Hawkes Process: A Neurally Self-Modulating
    Multivariate Point Process, https://arxiv.org/abs/1612.09328.

    The implementation uses code from https://github.com/yangalan123/anhp-andtt/blob/master/anhp/esm/thinning.py

    """

    def __init__(self, num_sample, num_exp, over_sample_rate, num_samples_boundary, dtime_max, patience_counter,
                 device):
        super(EventSampler, self).__init__()
        self.num_sample = num_sample
        self.num_exp = num_exp 
        self.over_sample_rate = over_sample_rate
        self.num_samples_boundary = num_samples_boundary
        self.dtime_max = dtime_max
        self.patience_counter = patience_counter
        self.device = device

    def compute_intensity_upper_bound(self, time_seq, time_delta_seq, event_seq, intensity_fn):
        """

        Args:
            time_seq: [batch_size, seq_len]
            time_delta_seq: [batch_size, seq_len]
            event_seq: [batch_size, seq_len]

        Returns:
            The upper bound of intensity at each event timestamp
            [batch_size, seq_len]

        """
        batch_size, seq_len = time_seq.size()

        # [1, 1, num_samples_boundary]
        time_for_bound_sampled = torch.linspace(start=0.0,
                                                end=1.0,
                                                steps=self.num_samples_boundary,
                                                device=self.device)[None, None, :]

        # [batch_size, seq_len, num_sample]
        dtime_for_bound_sampled = time_delta_seq[:, :, None] * time_for_bound_sampled

        # [batch_size, seq_len, num_sample, event_num]
        intensities_for_bound = intensity_fn(time_seq,
                                             time_delta_seq,
                                             event_seq,
                                             dtime_for_bound_sampled,
                                             max_steps=seq_len)

        # [batch_size, seq_len]
        bounds = intensities_for_bound.sum(dim=-1).max(dim=-1)[0] * self.over_sample_rate

        return bounds

    def sample_exp_distribution(self, sample_rate):
        """

        Args:
            sample_rate: [batch_size, seq_len]
            time_delta_seq: [batch_size, seq_len]

        Returns:
            exp_numbers: [batch_size, seq_len, num_sample, num_exp]

        """

        batch_size, seq_len = sample_rate.size()

        exp_numbers = torch.empty(size=[batch_size, seq_len, self.num_exp],
                                  dtype=torch.float32,
                                  device=self.device)

        exp_numbers.exponential_(1.0)
        exp_numbers = exp_numbers / sample_rate[:, :, None]

        return exp_numbers

    def sample_uniform_distribution(self, intensity_upper_bound):
        """

        Returns:
            unif_numbers: [batch_size, seq_len, num_sample, num_exp]

        """
        batch_size, seq_len = intensity_upper_bound.size()

        unif_numbers = torch.empty(size=[batch_size, seq_len, self.num_sample, self.num_exp],
                                   dtype=torch.float32, device=self.device)
        unif_numbers.uniform_(0.0, 1.0)

        return unif_numbers

    def sample_accept(self, unif_numbers, sample_rate, total_intensities):
        """

        Args:
            unif_numbers: [batch_size, max_len, num_sample, num_exp]
            sample_rate: [batch_size, max_len]
            total_intensities: [batch_size, seq_len, num_sample, num_exp]

        for each parallel draw, find its min criterion： if that < 1.0, the 1st (i.e. smallest) sampled time
        with cri < 1.0 is accepted; if none is accepted, use boundary / maxsampletime for that draw

        Returns:
            criterion, [batch_size, max_len, num_sample, num_exp]
            who_has_accepted_times, [batch_size, max_len, num_sample]

        """

        criterion = unif_numbers * sample_rate[:, :, None, None] / total_intensities

        min_cri_each_draw, _ = criterion.min(dim=-1)

        who_has_accepted_times = min_cri_each_draw < 1.0

        return criterion, who_has_accepted_times

    def draw_next_time_one_step(self, time_seq, time_delta_seq, event_seq, dtime_boundary,
                                intensity_fn):
        intensity_upper_bound = self.compute_intensity_upper_bound(time_seq,
                                                                   time_delta_seq,
                                                                   event_seq,
                                                                   intensity_fn)

        exp_numbers = self.sample_exp_distribution(intensity_upper_bound)

        intensities_at_sampled_times = intensity_fn(time_seq,
                                                    time_delta_seq,
                                                    event_seq,
                                                    exp_numbers,
                                                    max_steps=time_seq.size(1))

        total_intensities = intensities_at_sampled_times.sum(dim=-1)

        total_intensities = torch.tile(total_intensities[:, :, None, :], [1, 1, self.num_sample, 1])
        exp_numbers = torch.tile(exp_numbers[:, :, None, :], [1, 1, self.num_sample, 1])

        unif_numbers = self.sample_uniform_distribution(intensity_upper_bound)

        criterion, who_has_accepted_times = self.sample_accept(unif_numbers, intensity_upper_bound,
                                                               total_intensities)
        sampled_dtimes_accepted = exp_numbers.clone()

        sampled_dtimes_accepted[criterion >= 1.0] = exp_numbers.max() + 1.0

        accepted_times_each_draw, accepted_id_each_draw = sampled_dtimes_accepted.min(dim=-1)

        dtime_boundary = torch.tile(dtime_boundary[..., None], [1, 1, self.num_sample])

        res = torch.ones_like(dtime_boundary) * dtime_boundary

        weights = torch.ones_like(dtime_boundary)
        weights /= weights.sum(dim=-1, keepdim=True)

        res[who_has_accepted_times] = accepted_times_each_draw[who_has_accepted_times]
        who_not_accept = ~who_has_accepted_times

        who_reach_further = exp_numbers[..., -1] > dtime_boundary

        res[who_not_accept & who_reach_further] = exp_numbers[..., -1][who_not_accept & who_reach_further]

        return res, weights

