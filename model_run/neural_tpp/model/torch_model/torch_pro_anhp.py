import math

import torch
from torch import nn

from neural_tpp.model.torch_model.torch_baselayer import EncoderLayer, ConTEncoderLayer, MultiHeadAttention, PromptLayer, PrxfixPromptLayer, ContTPrxfixPromptLayer
from neural_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class PromptAttNHP(TorchBaseModel):
    """
    Reference: Continuous Transformer, ICLR 2022
    https://github.com/yangalan123/anhp-andtt/blob/master/anhp/model/xfmr_nhp_fast.py
    """

    def __init__(self, model_config):
        super(PromptAttNHP, self).__init__(model_config)
        self.d_model = model_config['hidden_size']
        self.d_time = model_config['time_emb_size']
        self.use_norm = model_config['use_ln']

        self.div_term = torch.exp(torch.arange(0, self.d_time, 2) * -(math.log(10000.0) / self.d_time)).reshape(1, 1,
                                                                                                                -1)
        self.n_layers = model_config['num_layers']
        self.n_head = model_config['num_heads']
        self.dropout = model_config['dropout']
        self.mc_num_sample_per_step = model_config['mc_num_sample_per_step']

        self.heads = []

        for i in range(self.n_head):
            self.heads.append(
                nn.ModuleList(
                    [ConTEncoderLayer(
                        self.d_model + self.d_time,
                        ContTPrxfixPromptLayer(1, self.d_model + self.d_time, self.d_model, self.dropout,
                                           output_linear=False),

                        use_residual=False,
                        dropout=self.dropout
                    )
                        for _ in range(self.n_layers)
                    ]
                )
            )
        

        self.heads = nn.ModuleList(self.heads)

        if self.use_norm:
            self.norm = nn.LayerNorm(self.d_model)
        self.inten_linear = nn.Linear(self.d_model * self.n_head, self.num_event_types_no_pad)
        self.softplus = nn.Softplus()
        self.layer_event_emb = nn.Linear(self.d_model + self.d_time, self.d_model)
        self.layer_intensity = nn.Sequential(self.inten_linear, self.softplus)
        self.eps = torch.finfo(torch.float32).eps

        self.model_to_device()
    
    def freeze_prompt_layer(self):
        # print("Freezing Prompt Layer .......")
        for head_i in range(self.n_head):
            for layer_i in range(self.n_layers):
                for p in self.heads[head_i][layer_i].self_attn.prompt_pool_k.parameters():
                    p.requires_grad = False
                for p in self.heads[head_i][layer_i].self_attn.prompt_pool_v.parameters():
                    p.requires_grad = False
    

    def freeze_backbone(self):
        # print("Freezing Backbone .......")
        pass

    def warm_prompt_layer(self):
        # print("Warming Prompt Layer .......")
        for head_i in range(self.n_head):
            for layer_i in range(self.n_layers):
                for p in self.heads[head_i][layer_i].self_attn.prompt_pool_k.parameters():
                    p.requires_grad = True
                for p in self.heads[head_i][layer_i].self_attn.prompt_pool_v.parameters():
                    p.requires_grad = True


    def compute_temporal_embedding(self, time):
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.d_time).to(time.device)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time.device)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)

        return pe

    def forward_pass(self, init_cur_layer, time_emb, est_time_emb, type_emb, sample_time_emb, event_emb, combined_mask):
        cur_layers = []
        total_prompt_key_loss = 0
        total_cde_loss = 0

        seq_len = event_emb.size(1)
        for head_i in range(self.n_head):
            cur_layer_ = init_cur_layer
            for layer_i in range(self.n_layers):
                layer_ = torch.cat([cur_layer_, sample_time_emb], dim=-1)
                _combined_input = torch.cat([event_emb, layer_], dim=1)
                enc_layer = self.heads[head_i][layer_i]

                enc_output, prompt_key_loss, cde_loss = enc_layer(_combined_input, time_emb, est_time_emb, type_emb, combined_mask)
                total_prompt_key_loss = total_prompt_key_loss + prompt_key_loss
                total_cde_loss = total_cde_loss +cde_loss

                _cur_layer_ = enc_output[:, seq_len:, :]
                cur_layer_ = torch.tanh(_cur_layer_) + cur_layer_

                event_emb = torch.cat([enc_output[:, :seq_len, :], time_emb], dim=-1)

                if self.use_norm:
                    cur_layer_ = self.norm(cur_layer_)
            cur_layers.append(cur_layer_)
        cur_layer_ = torch.cat(cur_layers, dim=-1)

        return cur_layer_, total_prompt_key_loss, total_cde_loss

    def seq_encoding(self, time_seqs, event_seqs):
        """

        Args:
            time_seqs: time seqs input, [batch_size, seq_len]
            event_seqs: event type seqs input, [batch_size, seq_len]

        Returns:

        """
        time_emb = self.compute_temporal_embedding(time_seqs)
        type_emb = torch.tanh(self.layer_type_emb(event_seqs.long()))
        event_emb = torch.cat([type_emb, time_emb], dim=-1)

        return event_emb, time_emb, type_emb

    def make_layer_mask(self, attention_mask):
        """

        Args:
            attention_mask: mask for attention operation, [batch_size, seq_len, seq_len]

        Returns:
            layer mask: mean to keep the current layer, the same size of attention mask
            a diagonal matrix, [batch_size, seq_len, seq_len]

        """
        layer_mask = (torch.eye(attention_mask.size(1)).to(self.device) < 1).unsqueeze(0).expand_as(attention_mask)
        return layer_mask

    def make_combined_att_mask(self, attention_mask, layer_mask):
        """

        Args:
            attention_mask: mask for attention operation, [batch_size, seq_len, seq_len]
            layer_mask: mask for other layers, [batch_size, seq_len, seq_len]

        Returns:
            combined_mask:  [batch_size, seq_len * 2, seq_len * 2]
        """
        combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)
        contextual_mask = torch.cat([attention_mask, torch.ones_like(layer_mask).to(self.device)], dim=-1)
        combined_mask = torch.cat([contextual_mask, combined_mask], dim=1)
        return combined_mask

    def forward(self, time_seqs, time_delta_seq, event_seqs, attention_mask, sample_times=None):
        event_emb, time_emb, type_emb = self.seq_encoding(time_seqs, event_seqs)
        init_cur_layer = torch.zeros_like(type_emb)
        layer_mask = self.make_layer_mask(attention_mask)
        if sample_times is None:
            sample_time_emb = time_emb
        else:
            sample_time_emb = self.compute_temporal_embedding(sample_times)
        combined_mask = self.make_combined_att_mask(attention_mask, layer_mask)

        estimated_inter_delta_time = torch.mean(time_delta_seq, dim = -1)
        estimated_inter_time = time_delta_seq[:,0] + estimated_inter_delta_time
        estimated_inter_time = estimated_inter_time.unsqueeze(-1)
        est_time_emb = self.compute_temporal_embedding(estimated_inter_time)


        '''
        cur_layer_ = self.forward_pass(init_cur_layer, time_emb, type_emb, sample_time_emb, event_emb, combined_mask)

        return cur_layer_
        '''
        cur_layer_, total_prompt_key_loss, total_cde_loss = self.forward_pass(init_cur_layer, time_emb, est_time_emb, type_emb, sample_time_emb, event_emb, combined_mask)

        return cur_layer_, total_prompt_key_loss, total_cde_loss

    def loglike_loss(self, batch):
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = batch
        enc_out, total_prompt_key_loss, total_cde_loss = self.forward(time_seq[:, :-1], time_delta_seq[:, :-1],event_seq[:, :-1], attention_mask[:, 1:, :-1], time_seq[:, 1:])
        lambda_at_event = self.layer_intensity(enc_out)

        temp_time = self.make_dtime_loss_samples(time_delta_seq[:, 1:])

        sample_times = temp_time + time_seq[:, :-1].unsqueeze(-1)

        lambda_t_sample = self.compute_intensities_at_sample_times(time_seq[:, :-1],
                                                                   time_delta_seq[:, :-1],
                                                                   event_seq[:, :-1],
                                                                   sample_times,
                                                                   attention_mask=attention_mask[:, 1:, :-1])

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seq[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        loss = - (event_ll - non_event_ll).sum()

        prompt_loss_alpha = 0.1
        loss = loss - prompt_loss_alpha * total_prompt_key_loss

        return loss, num_events

    def compute_states_at_sample_times(self,
                                       time_seqs,
                                       time_delta_seqs,
                                       type_seqs,
                                       attention_mask,
                                       sample_times):
        """

        Args:
            type_seqs: [batch_size, seq_len]
            time_seqs: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len, seq_len]
            sample_times: [batch_size, seq_len, num_samples]

        Returns:
            hidden states at all sampled times: [batch_size, seq_len, num_samples, hidden_size]

        """
        batch_size = type_seqs.size(0)
        seq_len = type_seqs.size(1)
        num_samples = sample_times.size(-1)

        sample_times = sample_times.permute((2, 0, 1))
        _sample_time = sample_times.reshape(num_samples * batch_size, -1)
        _types = type_seqs.expand(num_samples, -1, -1).reshape(num_samples * batch_size, -1)
        _times = time_seqs.expand(num_samples, -1, -1).reshape(num_samples * batch_size, -1)
        _times_delta = time_delta_seqs.expand(num_samples, -1, -1).reshape(num_samples * batch_size, -1)
        _attn_mask = attention_mask.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(num_samples * batch_size,
                                                                                         seq_len,
                                                                                         seq_len).to(self.device)
        encoder_output, total_prompt_key_loss, total_cde_loss = self.forward(_times,
                                      _times_delta,
                                      _types,
                                      _attn_mask,
                                      _sample_time)

        encoder_output = encoder_output.reshape(num_samples, batch_size, seq_len, -1)
        encoder_output = encoder_output.permute((1, 2, 0, 3))
        return encoder_output

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_times, **kwargs):
        """
        Args:
            time_seqs: [batch_size, seq_len]
            time_delta_seqs: [batch_size, seq_len]
            type_seqs: [batch_size, seq_len]
            sample_times: [batch_size, seq_len, num_samples]

        Returns:
            intensities at sample times: [batch_size, seq_len, num_samples, num_event_types]
        """
        attention_mask = kwargs.get('attention_mask', None)

        if attention_mask is None:
            batch_size, seq_len = time_seqs.size()
            attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1, -1).to(torch.bool)

        encoder_output = self.compute_states_at_sample_times(time_seqs, time_delta_seqs, type_seqs, attention_mask, sample_times)

        lambdas = self.layer_intensity(encoder_output)
        return lambdas
