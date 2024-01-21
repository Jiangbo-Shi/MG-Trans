# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
from os.path import join as pjoin

import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch.nn import ReLU
import models.configs as configs
from .model_utils import *

logger = logging.getLogger(__name__)
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()

        self.patch_embeddings = nn.Sequential(*[nn.Linear(config.input_size, config.hidden_size), nn.ReLU(), nn.Dropout(0.25)]) #
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, coords):
        x = x.unsqueeze(0)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patch_embeddings(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Part_GCN(torch.nn.Module):
    def __init__(self, config, edge_agg='spatial'):
        super(Part_GCN, self).__init__()
        self.edge_agg = edge_agg
        self.num_layers = 3

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(config.hidden_size, config.hidden_size, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(config.hidden_size, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', ckpt_grad=i % 3)
            self.layers.append(layer)
        self.conv1 = nn.Conv2d(config.hidden_size*3, config.hidden_size, kernel_size=1, stride=1)

    def forward(self, data):

        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        edge_index = edge_index.long()
        edge_attr = None
        batch = data.batch

        x = data.x
        x = self.layers[0].conv(x, edge_index, edge_attr)
        x_ = x
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)

        x = x_
        x = self.conv1(x.unsqueeze(-1).unsqueeze(-1))
        x = x.squeeze()

        return x, edge_index


class Block_graph(nn.Module):
    def __init__(self, config):
        super(Block_graph, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)
        self.gcn = Part_GCN(config)

    def forward(self, x, coord_s, threshold):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        token_graph = pt2graph(coord_s.squeeze(), x[:, 1:, :].squeeze(), threshold).to('cuda:0')
        graph_encoded, _ = self.gcn(token_graph)
        x[:, 0, :] = x[:, 0, :] + torch.mean(graph_encoded, dim=0).unsqueeze(0)

        return x, weights


class ATTShort(nn.Module):
    def __init__(self, config):
        super(ATTShort, self).__init__()
        self.sa_layer = Block(config)
        self.cross_layer = MultiheadAttention(embed_dim=config.hidden_size, num_heads=1)
        self.cross_layer_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.position_embeddings = nn.Parameter(torch.randn(1, 512+1, config.hidden_size))

    def forward(self, hidden_states, coord_s, sequence_length):
        hidden_states, attention_map = self.sa_layer(hidden_states)
        weight = attention_map[:, :, 0, 1:]
        weight = torch.mean(weight, dim=1)
        _, max_inx = weight.topk(sequence_length)
        part_inx = torch.unique(max_inx.reshape(-1)).unsqueeze(0) + 1
        part_hidden_states = hidden_states[0, part_inx[0, :]].unsqueeze(0)

        pooling_hidden_states, _ = self.cross_layer(part_hidden_states.permute(1, 0, 2),
                                                    hidden_states[:, 1:, :], hidden_states[:, 1:, :])

        pooling_hidden_states = pooling_hidden_states.permute(1, 0, 2)
        pooling_hidden_states = pooling_hidden_states + part_hidden_states
        pooling_hidden_states = self.cross_layer_norm(pooling_hidden_states)
        pooling_hidden_states = torch.cat((hidden_states[:, 0, :].unsqueeze(1), pooling_hidden_states), dim=1)

        coord_s = coord_s[part_inx[0, :]-1]

        return pooling_hidden_states, part_inx, weight, coord_s

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.layer1 = Block(config)
        self.layer2 = Block_graph(config)

    def forward(self, hidden_states, coord, threshold):
        attn_weights = []
        hidden_states, weights = self.layer1(hidden_states)
        attn_weights.append(weights)
        hidden_states, weights = self.layer2(hidden_states, coord, threshold)
        attn_weights.append(weights)

        return hidden_states, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, num_classes):
        super(Transformer, self).__init__()

        self.attshort = ATTShort(config)
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids, coord_s, x_l, coords_l, attn_mask):
        sequence_length = 512
        if(input_ids.shape[0] < sequence_length):
            num = sequence_length / input_ids.shape[0]
            new_input_ids = input_ids
            new_coords = coord_s
            new_attn_mask = attn_mask
            for i in range(int(num)):
                new_input_ids = torch.cat((new_input_ids, input_ids), dim=0)
                new_coords = torch.cat((new_coords, coord_s), dim=0)
                new_attn_mask = torch.cat((new_attn_mask, attn_mask), dim=0)
            input_ids = new_input_ids[:sequence_length, :]
            coord_s = new_coords[:sequence_length, :]
            attn_mask = new_attn_mask[:sequence_length, :]
        embedding_output = self.embeddings(input_ids, coord_s)

        # PAM
        hidden_states, part_idx, AM, coord_s = self.attshort(embedding_output, coord_s, sequence_length)
        all_tokens, _ = self.encoder(hidden_states, coord_s, 2048)
        
        # high scale
        x_l_index = np.nonzero(attn_mask[part_idx-1][0])[:, 1]
        x_l_part = x_l[x_l_index]
        coords_l = coords_l[x_l_index]
        embedding_output_l = self.embeddings(x_l_part, coords_l)
        all_tokens_l, attention_map_l = self.encoder(embedding_output_l, coords_l, 2048)
        reverse_mask = np.nonzero(attn_mask[part_idx-1][0])[:, 0]

        return all_tokens, AM, all_tokens_l, reverse_mask, attention_map_l, hidden_states, embedding_output_l, all_tokens


class MG_Trans_main(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False):
        super(MG_Trans_main, self).__init__()

        self.transformer = Transformer(config, img_size, num_classes)
        self.IB_layer = Linear(config.hidden_size*2, config.hidden_size)
        self.head = Linear(config.hidden_size, num_classes)

        self.head_low = Linear(config.hidden_size, num_classes)
        self.head_high = Linear(config.hidden_size, num_classes)

        self.loss_mse = nn.MSELoss()
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_l1 = nn.L1Loss()

    def forward(self, x_s, coord_s, x_l, coords_l, attn_msk, label, attention_only=False, labels=None):
        all_tokens, attention_map, all_tokens_l, reverse_mask, attention_map_l, x1, x2, AM = self.transformer(x_s, coord_s, x_l, coords_l, attn_msk)
        fusion_tokens = torch.cat((all_tokens[:, 0], all_tokens_l[:, 0]), dim=1)
        Z = self.IB_layer(fusion_tokens)

        logits = self.head(Z)

        # cross-entropy
        loss_ce = self.loss_ce(logits, label)

        # glocal semantic loss
        logits_low = self.head_low(all_tokens[:, 0])
        logits_high = self.head_high(all_tokens_l[:, 0])
        delta_logits = torch.norm(logits_high-logits_low, p=1, dim=0).unsqueeze(0)
        loss_global = self.loss_ce(delta_logits, label)

        # total loss
        loss = loss_ce + 1e-4 * loss_global

        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]

        x1 = x1[:, 0, :] + torch.mean(x1[:, 1:, :], dim=1)
        x2 = x2[:, 0, :] + torch.mean(x2[:, 1:, :], dim=1)

        z1 = all_tokens[:, 0]
        z2 = all_tokens_l[:, 0]
        out = logits

        return logits, Y_prob, Y_hat, attention_map, loss, x1, x2, z1, z2, out

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.part_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.part_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

CONFIGS = {
    'MG_Trans': configs.get_config(),
}