import math
import torch
from torch import nn


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        seq_len = mask.shape[-1]
        scores[:,:,:seq_len,:seq_len] = scores[:,:,:seq_len,:seq_len].masked_fill(mask > 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.d_model = d_model
        self.output_linear = output_linear

        if output_linear:
            self.linears = nn.ModuleList(
                [nn.Linear(d_input, d_model) for _ in range(3)] + [nn.Linear(d_model, d_model), ])
        else:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            return self.linears[-1](x)
        else:
            return x


class SublayerConnection(nn.Module):
    # used for residual connection
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward=None, use_residual=False, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual
        if use_residual:
            self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask):
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            return self.self_attn(x, x, x, mask)


class PromptPool(nn.Module):
    def __init__(self, length=5, embed_dim=32, embedding_key='mean', prompt_init='uniform',  
                 pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()
    
        self.length = length
        self.embed_dim = embed_dim
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        prompt_pool_shape = (pool_size, length, embed_dim)
        if prompt_init == 'zero':
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == 'uniform':
            self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
            nn.init.uniform_(self.prompt, -1, 1)
        
        key_shape = (pool_size, embed_dim)
        if prompt_key_init == 'zero':
            self.prompt_key = nn.Parameter(torch.zeros(key_shape))
        elif prompt_key_init == 'uniform':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.prompt_key, -1, 1)
    

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()

        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            if cls_features is None:
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            else:
                x_embed_mean = cls_features
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")

        prompt_norm = self.l2_normalize(self.prompt_key, dim=1)
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)


        similarity = torch.matmul(x_embed_norm, prompt_norm.t())
            
        if prompt_mask is None:
            _, idx = torch.topk(similarity, k=self.top_k, dim=1)
            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k)
                major_prompt_id = prompt_id[major_idx] 
                idx = major_prompt_id.expand(x_embed.shape[0], -1)
        else:
            idx = prompt_mask

        batched_prompt_raw = self.prompt[idx]
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)

        out['prompt_idx'] = idx

        out['prompt_norm'] = prompt_norm
        out['x_embed_norm'] = x_embed_norm
        out['similarity'] = similarity

        batched_key_norm = prompt_norm[idx]
        out['selected_key'] = batched_key_norm
        x_embed_norm = x_embed_norm.unsqueeze(1)
        sim = batched_key_norm * x_embed_norm 
        reduce_sim = torch.sum(sim) / x_embed.shape[0]

        out['reduce_sim'] = reduce_sim

        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out

class PromptLayer(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False, 
                 prompt_length = 5, embed_dim = 48, embedding_key = 'mean', prompt_init = 'uniform',pool_size = 10, top_k = 5, batchwise_prompt = True, prompt_key_init = 'uniform'):
        super(PromptLayer, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.d_model = d_model
        self.output_linear = output_linear

        if output_linear:
            self.linears = nn.ModuleList(
                [nn.Linear(d_input, d_model) for _ in range(3)] + [nn.Linear(d_model, d_model), ])
        else:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)
        
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.prompt_key_init = prompt_key_init
        self.prompt_pool_k = PromptPool(length=self.prompt_length, embed_dim=self.embed_dim, embedding_key=self.embedding_key, prompt_init=self.prompt_init,
                    pool_size=self.pool_size, top_k=self.top_k, batchwise_prompt=self.batchwise_prompt,
                    prompt_key_init=self.prompt_key_init)
        self.prompt_pool_q = PromptPool(length=self.prompt_length, embed_dim=self.embed_dim, embedding_key=self.embedding_key, prompt_init=self.prompt_init,
                    pool_size=self.pool_size, top_k=self.top_k, batchwise_prompt=self.batchwise_prompt,
                    prompt_key_init=self.prompt_key_init)
        self.prompt_pool_v = PromptPool(length=self.prompt_length, embed_dim=self.embed_dim, embedding_key=self.embedding_key, prompt_init=self.prompt_init,
                    pool_size=self.pool_size, top_k=self.top_k, batchwise_prompt=self.batchwise_prompt,
                    prompt_key_init=self.prompt_key_init)

    def forward(self, query, key, value, mask):

        prompt_mask = None
        cls_features = None
        res_k = self.prompt_pool_k(key, prompt_mask=prompt_mask, cls_features=cls_features)
        res_q = self.prompt_pool_q(query, prompt_mask=prompt_mask, cls_features=cls_features)
        res_v = self.prompt_pool_v(value, prompt_mask=prompt_mask, cls_features=cls_features)

        key = res_k['prompted_embedding']
        query = res_q['prompted_embedding']
        value = res_v['prompted_embedding']
        prompt_len = res_k['total_prompt_len']
        

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        x = x[:,prompt_len:,:]

        if self.output_linear:
            return self.linears[-1](x)
        else:
            return x


def prefix_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        seq_len = mask.shape[-1]
        scores[:,:,:seq_len,:seq_len] = scores[:,:,:seq_len,:seq_len].masked_fill(mask > 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def prefix_attention_nhp(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        seq_len = mask.shape[-1]
        scores[:,:,:seq_len,:seq_len] = scores[:,:,:seq_len,:seq_len].masked_fill(mask > 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class PrxfixPromptLayer(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False, 
                 prompt_length = 5, embed_dim = 64, embedding_key = 'mean', prompt_init = 'uniform', pool_size = 10, top_k = 5, batchwise_prompt = True, prompt_key_init = 'uniform'):
        super(PrxfixPromptLayer, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.d_model = d_model
        self.output_linear = output_linear

        if output_linear:
            self.linears = nn.ModuleList(
                [nn.Linear(d_input, d_model) for _ in range(3)] + [nn.Linear(d_model, d_model), ])
        else:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)
        
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.prompt_key_init = prompt_key_init
        self.prompt_pool_k = PromptPool(length=self.prompt_length, embed_dim=self.embed_dim, embedding_key=self.embedding_key, prompt_init=self.prompt_init,
                    pool_size=self.pool_size, top_k=self.top_k, batchwise_prompt=self.batchwise_prompt,
                    prompt_key_init=self.prompt_key_init)
        self.prompt_pool_v = PromptPool(length=self.prompt_length, embed_dim=self.embed_dim, embedding_key=self.embedding_key, prompt_init=self.prompt_init,
                    pool_size=self.pool_size, top_k=self.top_k, batchwise_prompt=self.batchwise_prompt,
                    prompt_key_init=self.prompt_key_init)

    def forward(self, query, key, value, mask):
        prompt_mask = None
        cls_features = None
        res_k = self.prompt_pool_k(key, prompt_mask=prompt_mask, cls_features=cls_features)
        res_v = self.prompt_pool_v(value, prompt_mask=prompt_mask, cls_features=cls_features)

        key = res_k['prompted_embedding']
        value = res_v['prompted_embedding']
        prompt_len = res_k['total_prompt_len']

        prompt_key_loss = res_k['reduce_sim'] + res_v['reduce_sim']
        

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        x, attn_weight = prefix_attention_nhp(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            return self.linears[-1](x)
        else:
            return x, prompt_key_loss


class ContTPromptPool(nn.Module):
    def __init__(self, length=5, embed_dim=32, embedding_key='mean', prompt_init='uniform',  
                 pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 time_cde = None, type_embed_dim = 32 ,time_embed_dim = 16, est_time_type = 'MEAN'):
        super().__init__()
    
        self.length = length
        self.embed_dim = embed_dim
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.est_time_type = est_time_type

        self.time_cde = time_cde
        self.type_embed_dim = type_embed_dim
        self.time_embed_dim = time_embed_dim

        type_prompt_pool_shape = (pool_size, length, type_embed_dim)
        if prompt_init == 'zero':
            self.type_prompt = nn.Parameter(torch.zeros(type_prompt_pool_shape))
        elif prompt_init == 'uniform':
            self.type_prompt = nn.Parameter(torch.randn(type_prompt_pool_shape))
            nn.init.uniform_(self.type_prompt, -1, 1)
        self.time_prompt = None
        self.prompt = None
        
        key_shape = (pool_size, embed_dim)
        if prompt_key_init == 'zero':
            self.prompt_key = nn.Parameter(torch.zeros(key_shape))
        elif prompt_key_init == 'uniform':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.prompt_key, -1, 1)
    

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    

    def forward(self, x_embed, est_time_emb, prompt_mask=None, cls_features=None):
        out = dict()
        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            if cls_features is None:
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            else:
                x_embed_mean = cls_features
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")

        prompt_norm = self.l2_normalize(self.prompt_key, dim=1)
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)


        similarity = torch.matmul(x_embed_norm, prompt_norm.t())
            
        if prompt_mask is None:
            _, idx = torch.topk(similarity, k=self.top_k, dim=1)
            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k)
                major_prompt_id = prompt_id[major_idx]
                idx = major_prompt_id.expand(x_embed.shape[0], -1)
        else:
            idx = prompt_mask

        if self.est_time_type == 'CDE':
            self.time_prompt = self.time_cde(self.type_prompt)
        else:
            self.time_prompt = est_time_emb

        self.prompt = self.type_prompt


        batched_prompt_raw = self.prompt[idx]
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)

        self.time_prompt = est_time_emb.expand(-1, batched_prompt.size()[-2], -1)
        batched_prompt = torch.cat([batched_prompt, self.time_prompt], dim=-1)

        out['prompt_idx'] = idx

        out['prompt_norm'] = prompt_norm
        out['x_embed_norm'] = x_embed_norm
        out['similarity'] = similarity

        batched_key_norm = prompt_norm[idx]
        out['selected_key'] = batched_key_norm
        x_embed_norm = x_embed_norm.unsqueeze(1)
        sim = batched_key_norm * x_embed_norm
        reduce_sim = torch.sum(sim) / x_embed.shape[0]

        out['reduce_sim'] = reduce_sim

        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out

class ConTEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward=None, use_residual=False, dropout=0.1):
        super(ConTEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual
        if use_residual:
            self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, time_emb, est_time_emb, type_emb, mask):
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            return self.self_attn(x, time_emb, est_time_emb, type_emb, mask)

class ContTPrxfixPromptLayer(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False, type_embed_dim = 32, time_embed_dim = 16, 
                 prompt_length = 5, embedding_key = 'mean', prompt_init = 'uniform', pool_size = 10, top_k = 5, batchwise_prompt = True, prompt_key_init = 'uniform'):
        super(ContTPrxfixPromptLayer, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.d_model = d_model
        self.output_linear = output_linear

        if output_linear:
            self.linears = nn.ModuleList(
                [nn.Linear(d_input, d_model) for _ in range(3)] + [nn.Linear(d_model, d_model), ])
        else:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)

        self.type_embed_dim = type_embed_dim
        self.time_embed_dim = time_embed_dim
        self.time_cde = nn.Linear(self.type_embed_dim, self.time_embed_dim, bias=True)
        self.cde_loss_fn = torch.nn.MSELoss(reduction='mean')


        self.prompt_length = prompt_length
        self.embed_dim = self.type_embed_dim + self.time_embed_dim
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.prompt_key_init = prompt_key_init
        self.prompt_pool_k = ContTPromptPool(length=self.prompt_length, embed_dim=self.embed_dim, embedding_key=self.embedding_key, prompt_init=self.prompt_init,
                    pool_size=self.pool_size, top_k=self.top_k, batchwise_prompt=self.batchwise_prompt, prompt_key_init=self.prompt_key_init,
                    time_cde = self.time_cde, type_embed_dim = self.type_embed_dim  ,time_embed_dim = self.time_embed_dim)
        self.prompt_pool_v = ContTPromptPool(length=self.prompt_length, embed_dim=self.embed_dim, embedding_key=self.embedding_key, prompt_init=self.prompt_init,
                    pool_size=self.pool_size, top_k=self.top_k, batchwise_prompt=self.batchwise_prompt, prompt_key_init=self.prompt_key_init,
                    time_cde = self.time_cde, type_embed_dim = self.type_embed_dim  ,time_embed_dim = self.time_embed_dim)
    

    def forward(self, x, time_emb, est_time_emb, type_emb, mask):

        time_emb_est = self.time_cde(type_emb)
        cde_loss = self.cde_loss_fn(time_emb_est, time_emb)

        query = x
        key = x
        value = x
        prompt_mask = None
        cls_features = None
        res_k = self.prompt_pool_k(key, est_time_emb, prompt_mask=prompt_mask, cls_features=cls_features)
        res_v = self.prompt_pool_v(value, est_time_emb, prompt_mask=prompt_mask, cls_features=cls_features)

        key = res_k['prompted_embedding']
        value = res_v['prompted_embedding']
        prompt_len = res_k['total_prompt_len']

        prompt_key_loss = res_k['reduce_sim'] + res_v['reduce_sim']

        

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        x, attn_weight = prefix_attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            return self.linears[-1](x), prompt_key_loss
        else:
            return x, prompt_key_loss, cde_loss

