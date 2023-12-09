import torch
from torch.utils.data import Dataset, DataLoader
from neural_tpp.preprocess.input import EventSeq


class TorchTPPDataset(Dataset):
    def __init__(self, data_config, **kwargs):
        self.data_config = data_config
        self.kwargs = kwargs
        self.event_seq_cls = EventSeq(event_num=data_config['event_num'],
                                      add_bos=data_config.get('add_bos', False),
                                      add_eos=data_config.get('add_eos', False),
                                      data_dir=data_config.get('data_dir', None),
                                      eos_elapse=data_config.get('eos_elapse', False),
                                      pad_end=data_config.get('pad_end', True),
                                      source_data=data_config.get('source_data', None))
        self.time_seq, self.time_delta_seq, self.event_seq = self.event_seq_cls.build()
        # get the max len of the sequence
        self.max_len = max([len(x) for x in self.time_seq])

    def __len__(self):
        """

        Returns: length of the dataset

        """
        assert len(self.time_seq) == len(self.event_seq) and len(self.time_delta_seq) == len(self.event_seq), \
            f"Inconsistent lengths for data! time_seq_len:{len(self.time_seq)}, event_len: " \
            f"{len(self.event_seq)}, time_delta_seq_len: {len(self.time_delta_seq)}"
        return len(self.event_seq)

    def __getitem__(self, idx):
        """

        Args:
            idx: iteration index

        Returns:
            time_seq, time_delta_seq and event_seq element

        """
        return self.time_seq[idx], self.time_delta_seq[idx], self.event_seq[idx]

    def collate_fn(self, batch):
        """

        Args:
            batch: batch sequence data

        Returns:
            batch tensors of time_seq, time_delta_seq, event_seq,
            batch_non_pad_mask, attention_mask, type_mask

        """
        time_seq, time_delta_seq, event_seq = list(zip(*batch))

        # use float64 to avoid precision loss during conversion from numpy.array to torch.tensor
        time_seq = torch.tensor(self.event_seq_cls.batch_pad_sequence(time_seq),
                                dtype=torch.float32)
        time_delta_seq = torch.tensor(self.event_seq_cls.batch_pad_sequence(time_delta_seq),
                                      dtype=torch.float32)
        event_seq = torch.tensor(self.event_seq_cls.batch_pad_sequence(event_seq),
                                 dtype=torch.long)

        batch_non_pad_mask, attention_mask = self.event_seq_cls.batch_attn_mask_for_pad_sequence(
            event_seq)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        type_mask = torch.tensor(self.event_seq_cls.batch_type_mask(event_seq), dtype=torch.bool)

        return time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask

    @property
    def num_event_types_pad(self):
        """

        Returns: num event types with padding

        """
        return self.event_seq_cls.event_num_with_pad

    @property
    def num_event_types_no_pad(self):
        """

        Returns: num event types without padding

        """

        return self.event_seq_cls.event_num

    @property
    def event_pad_index(self):
        """

        Returns: pad index for event sequence

        """

        return self.event_seq_cls.pad_index


def create_torch_dataloader(dataset, batch_size, **kwargs):
    """

    Args:
        dataset: TorchTPPDataset object
        batch_size: batch size to load the data
        **kwargs: optional parameters, e.g., shuffle, num_workers

    Returns:
        torch.DataLoader object

    """
    return DataLoader(dataset,
                      batch_size=batch_size,
                      collate_fn=dataset.collate_fn,
                      shuffle=kwargs.get('shuffle', True))
