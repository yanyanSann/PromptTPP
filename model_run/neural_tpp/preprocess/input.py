# Copyright (c) Ant Group and its affiliates.

import pickle

import numpy as np

from neural_tpp.utils import py_assert


class Input:
    """
    Base class for building event sequence from source_2
    """

    def __init__(self, event_num, add_bos, add_eos, eos_elapse):
        """
        Args:
            event_num: int, total number of events (no padding)
            add_bos: bool, whether to add BOS to event sequences
            add_eos: bool, whether to add EOS to event sequences
            eos_elapse: how much time we should wait after the last event to give EOS mark
        """
        self.event_num = event_num
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.eos_elapse = eos_elapse
        self.pad_index = self.event_num
        self.bos_index = self.event_num + 1
        self.eos_index = self.event_num + 2

    @staticmethod
    def read_from_pickle(data_dir):
        """
        Args:
            data_dir: dir of the pickle file

        Returns:
            data loaded from the pickle file

        """
        try:
            data = pickle.load(data_dir, encoding='latin-1')
        except Exception:
            data = pickle.load(data_dir)
        return data

    @staticmethod
    def insert_bos_sequence(seqs, bos_index):
        """

        Args:
            seqs: list of sequences with variational length
            bos_index: a value that is inserted as the beginning of sequence (BOS) event for event sequence

        Returns:
            list  of sequences with BOS inserted

        """
        return [[bos_index, ] + seq for seq in seqs]

    @staticmethod
    def insert_eos_sequence(seqs, eos_index):
        """

        Args:
            seqs: list of sequences with variational length
            eos_index: a value that is inserted as the end of sequence (EOS) event for event sequence

        Returns:
            list  of sequences with EOS inserted

        """
        return [seq + [eos_index, ] for seq in seqs]


class EventSeq(Input):
    def __init__(self, event_num, add_bos, add_eos, data_dir, eos_elapse, pad_end, source_data=None):
        super(EventSeq, self).__init__(event_num, add_bos, add_eos, eos_elapse)
        py_assert(data_dir is not None or source_data is not None,
                  ValueError,
                  'data_dir or source_data must not be None simultaneously')
        if source_data is None:
            source_data = self.read_from_pickle(data_dir)
        self.time_seqs = [[x["time_since_start"] for x in seq] for seq in source_data]
        self.type_seqs = [[x["type_event"] for x in seq] for seq in source_data]
        self.time_delta_seqs = [[x["time_since_last_event"] for x in seq] for seq in source_data]
        self.pad_end = pad_end
        # at least include [PAD]
        self.event_num_with_pad = self.event_num + 1
        self.eos_elapse = eos_elapse

        py_assert(max([max(seq) for seq in self.type_seqs]) + 1 <= self.event_num,
                  ValueError,
                  "there are more event than specified?")

    def build(self):
        """

        Returns: numpy.array, sequences that are added BOS, EOS (optional).

        For simplicity, we pad the time_seq, event_seq and time_delta_seq with the same pad index

        """

        if self.add_bos:
            self.time_seqs = self.insert_bos_sequence(self.time_seqs, 0)
            self.type_seqs = self.insert_bos_sequence(self.type_seqs, self.bos_index)
            self.time_delta_seqs = self.insert_bos_sequence(self.time_delta_seqs, 0)
            self.event_num_with_pad += 1
        if self.add_eos:
            self.time_seqs = [seq + [seq[-1] + self.eos_elapse, ] for seq in self.time_seqs]
            self.type_seqs = self.insert_bos_sequence(self.type_seqs, self.eos_index)
            self.time_delta_seqs = [seq + [self.eos_elapse, ] for seq in self.time_delta_seqs]
            self.event_num_with_pad += 1

        return self.time_seqs, self.time_delta_seqs, self.type_seqs

    def batch_pad_sequence(self, seqs, pad_index=None, max_len=None):
        """
        Args:
            seqs: list of sequences with variational length
            pad_index: optional, a value that used to pad the sequences. If None, then the pad index
            is set to be the event_num_with_pad
            max_len: optional, the maximum length of the sequence after padding. If None, then the
            length is set to be the max length of all input sequences.
            pad_at_end: optional, whether to pad the sequnce at the end. If False, the sequence is pad at the beginning

        Returns:
            a numpy array of padded sequence


        Example:
        ```python
        seqs = [[0, 1], [3, 4, 5]]
        pad_sequence(seqs, 100)
        >>> [[0, 1, 100], [3, 4, 5]]

        pad_sequence(seqs, 100, max_len=5)
        >>> [[0, 1, 100, 100, 100], [3, 4, 5, 100, 100]]
        ```

        """
        pad_index = self.pad_index if pad_index is None else pad_index
        if max_len is None:
            max_len = max(len(seq) for seq in seqs)
        if self.pad_end:
            pad_seq = np.array([seq + [pad_index] * (max_len - len(seq)) for seq in seqs], np.float64)
        else:
            pad_seq = np.array([[pad_index] * (max_len - len(seq)) + seq for seq in seqs], np.float64)
        return pad_seq

    def batch_attn_mask_for_pad_sequence(self, pad_seqs, pad_index=None):
        """

        Args:
            pad_seqs: list of sequences that have been padded with fixed length
            pad_index: optional, a value that used to pad the sequences. If None, then the pad index
            is set to be the event_num_with_pad

        Returns:
            a bool matrix of the same size of input, denoting the masks of the sequence (True: non mask, False: mask)


        Example:
        ```python
        seqs = [[ 1,  6,  0,  7, 12, 12],
        [ 1,  0,  5,  1, 10,  9]]
        batch_attn_mask_for_pad_sequence(seqs, pad_index=12)
        >>>
            batch_non_pad_mask
            ([[ True,  True,  True,  True, False, False],
            [ True,  True,  True,  True,  True,  True]])
            attention_mask
            [[[ True  True  True  True  True  True]
              [False  True  True  True  True  True]
              [False False  True  True  True  True]
              [False False False  True  True  True]
              [False False False False  True  True]
              [False False False False  True  True]]

             [[ True  True  True  True  True  True]
              [False  True  True  True  True  True]
              [False False  True  True  True  True]
              [False False False  True  True  True]
              [False False False False  True  True]
              [False False False False False  True]]]
        ```


        """

        pad_index = self.pad_index if pad_index is None else pad_index
        seq_num, seq_len = pad_seqs.shape

        # [batch_size, seq_len]
        seq_pad_mask = pad_seqs == pad_index

        # [batch_size, seq_len, seq_len]
        attention_key_pad_mask = np.tile(seq_pad_mask[:, None, :], (1, seq_len, 1))
        subsequent_mask = np.tile(np.triu(np.ones((seq_len, seq_len), dtype=bool), k=0)[None, :, :], (seq_num, 1, 1))

        attention_mask = subsequent_mask | attention_key_pad_mask

        return ~seq_pad_mask, attention_mask

    def batch_type_mask(self, event_seq):
        """

        Args:
            event_seq: a list of sequence events with equal length (i.e., padded sequence)

        Returns:
            a 3-dim matrix, where the last dim (one-hot vector) indicates the type of event

        """
        type_mask = np.zeros([*event_seq.shape, self.event_num])
        for i in range(self.event_num):
            type_mask[:, :, i] = event_seq == i

        return type_mask
