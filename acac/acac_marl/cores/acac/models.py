import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from acac.acac_marl.cores.acac.transformer_model import GPT2Model
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import math
from acac.acac_marl.cores.acac.popart import PopArt

def positionalencoding1d(length,d_model):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def Linear(input_dim, output_dim, act_fn='leaky_relu', init_weight_uniform=True):
    gain = torch.nn.init.calculate_gain(act_fn)
    fc = torch.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.xavier_normal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc


"""
====================================================
Agent Centric Model
====================================================
"""

class GRUEncoder(nn.Module):

    def __init__(self, input_dim, mlp_layer_size=[32,32], rnn_layer_size=32, use_time_emb=False, time_emb_alg = 'sinu', time_emb_dim=4, max_timestep=200):
        super(GRUEncoder, self).__init__()
        self.use_time_emb = use_time_emb
        if use_time_emb:
            input_dim = input_dim+int(time_emb_dim)

        self.fc1 = Linear(input_dim, mlp_layer_size[0], act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        self.gru = nn.GRU(mlp_layer_size[0], hidden_size=rnn_layer_size, num_layers=1, batch_first=True)

    def forward(self, x, h=None, eps=0.0, test_mode=False):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x, h = self.gru(x, h)
        return x, h


class AgentCentricGRUActor(nn.Module):

    def __init__(self, input_dim, output_dim, mlp_layer_size=[32,32], rnn_layer_size=32, encoder=None, use_time_emb=False, time_emb_alg='sinu', time_emb_dim=4, max_timestep=200, use_ln=False):
        super(AgentCentricGRUActor, self).__init__()
        self.use_time_emb = use_time_emb
        self.use_ln = use_ln
        if use_time_emb:
            if time_emb_alg == 'sinu':
                self.time_embedding = nn.Embedding.from_pretrained(positionalencoding1d(max_timestep+5,int(time_emb_dim)))
            elif time_emb_alg =='learn':
                self.time_embedding = nn.Embedding(max_timestep+5,int(time_emb_dim))
            else :
                self.time_embedding =None
            input_dim = input_dim + int(time_emb_dim)
        if encoder is None:
            self.encoder = GRUEncoder(input_dim, mlp_layer_size, rnn_layer_size)
        else:
            self.encoder = encoder
        assert isinstance(self.encoder, GRUEncoder), "self.encoder should be a GRUEncoder!"

        self.fc3 = Linear(rnn_layer_size, mlp_layer_size[1], act_fn='leaky_relu')
        if use_ln:
            self.ln = nn.LayerNorm(mlp_layer_size[1])
        self.fc4 = Linear(mlp_layer_size[1], output_dim, act_fn='linear')

    def forward(self, x, h=None, eps=0.0, test_mode=False, time_emb=None):
        if time_emb is not None :
            time_emb = self.time_embedding(time_emb)
            x = torch.concat([x,time_emb],dim=-1)
        x, h = self.encoder(x, h)
        x = F.leaky_relu(self.fc3(x))
        if self.use_ln:
            x = self.ln(x)
        x = self.fc4(x)

        action_logits = F.log_softmax(x, dim=-1)

        if not test_mode:
            logits_1 = action_logits + np.log(1-eps)
            logits_2 = torch.full_like(action_logits, np.log(eps)-np.log(action_logits.size(-1)))
            logits = torch.stack([logits_1, logits_2])
            action_logits = torch.logsumexp(logits,axis=0)

        return action_logits, h

class AgentCentricGRUCritic(nn.Module):

    def __init__(self, input_dim, output_dim=1, mlp_layer_size=[32,32], rnn_layer_size=32, n_agent=3, encoders=None, use_attention=True, cc_n_head=2, value_head="concat",use_time_emb=False,time_emb_alg = 'sinu', time_emb_dim=4, max_timestep=200, use_popart=False, duplicate=False):
        super(AgentCentricGRUCritic, self).__init__()
        self.use_attention = use_attention
        self.value_head = value_head
        self.use_time_emb=use_time_emb
        self.use_popart = use_popart
        self.output_dim = output_dim
        if self.use_time_emb :
            if time_emb_alg == 'sinu' :
                self.time_embedding = nn.Embedding.from_pretrained(positionalencoding1d(max_timestep+5,int(time_emb_dim)))
            elif time_emb_alg =='learn' :
                self.time_embedding = nn.Embedding(max_timestep+5,int(time_emb_dim))
            else :
                self.time_embedding =None
            input_dim = np.array(input_dim)+int(time_emb_dim)
        if encoders is None:
            # NOTE: each agent obs dim can differ
            if type(input_dim) is int or type(input_dim) == np.int64:
                input_dim =[input_dim for _ in range(n_agent)]
            self.encoders = nn.ModuleList([GRUEncoder(input_dim[i], mlp_layer_size, rnn_layer_size) for i in range(n_agent)])
        else:
            self.encoders = nn.ModuleList(encoders)
        self.n_agent = n_agent
        self.duplicate = duplicate
        if self.use_attention:
            config = transformers.GPT2Config(
                vocab_size=1,  # doesn't matter -- we don't use the vocab
                n_embd=rnn_layer_size,
                n_layer=1,
                n_head=cc_n_head,
                n_positions=1024,
            )
            self.centralized_transformer = GPT2Model(config, use_causal_mask=False)
            if value_head == "concat":
                if self.use_popart:
                    self.value = PopArt(rnn_layer_size * n_agent, output_dim)
                else:
                    self.value = Linear(rnn_layer_size * n_agent, output_dim, act_fn='linear')
            elif value_head in ["share_linear_add", "add_linear", "mean_linear", "max_linear", "min_linear"]:
                if self.use_popart:
                    self.value = PopArt(rnn_layer_size, output_dim)
                else:
                    self.value = Linear(rnn_layer_size, output_dim, act_fn='linear')
            else:
                raise NotImplementedError
        else:
            self.fc = Linear(rnn_layer_size * n_agent, mlp_layer_size[1], act_fn='leaky_relu')
            if self.use_popart:
                self.value = PopArt(mlp_layer_size[1], output_dim)
            else:
                self.value = Linear(mlp_layer_size[1], output_dim, act_fn='linear')

    def recover_joint_feature(self, x_agent, v):
        # x_agent should be a tensor of size n_batch x max_squ_epi_length x dim_feature
        # v should be a tensor of size n_batch x trace_len
        idx = (torch.cumsum(v, dim=1) - 1).to(torch.int64).unsqueeze(-1).tile((1, 1, x_agent.size()[2])) # n_batch x trace_len x dim_feature
        j_x = torch.gather(x_agent, 1, idx) # n_batch x trace_len x dim_feature
        return j_x

    def forward(self, x, v, h=None, time_emb=None):
        # x should be a tensor of size n_batch x max_j_squ_epi_length (=trace_len) x (dim_obs (+ dim_action) * n_agent)
        # v should be a tensor of size n_batch x trace_len x n_agent
        # h should be a tensor of size n_hidden_layer x (dim_h * n_agent)
        
        assert x.size()[0] == v.size()[0], f'The length of x ({x.size()[0]}) should be matched to the length of v ({v.size()[0]}). Both values are number of batch.'
        assert x.size()[1] == v.size()[1], f'The width of x ({x.size()[1]}) should be matched to the width of v ({v.size()[1]}). Both values are the maximum squeezed joint episode length.'
        n_batch, trace_len, n_agent = v.size()[0], v.size()[1], self.n_agent
        device = x.device
        # For agent centric transformer
        x = x.reshape((n_batch, trace_len, n_agent, -1)).permute(2,0,1,3).contiguous() # n_agent x n_batch x trace_len x dim_obs
        if time_emb is not None :
            time_emb = self.time_embedding(time_emb.repeat(n_agent,1,1))
            x = torch.concat([x,time_emb],dim=-1)
        v = v.permute(2,0,1).contiguous() # n_agent x n_batch x trace_len

            
        j_v = torch.amax(v, dim=0) # n_batch x trace_len
        if h is not None:
            n_layers = h.size()[0]
            h = h.reshape((n_layers, n_agent, -1)).permute(1,0,2).contiguous() # n_agent x n_layers x dim_h
        else:
            n_layers = 0

        feature = []
        h_next = []
        for agent_idx in range(n_agent):
            if self.duplicate:
                v_agent = v.sum(0).bool()
            else:
                v_agent = v[agent_idx]
            squ_epi_len_agent = v_agent.sum(1)
            x_agent = torch.split_with_sizes(x[agent_idx][v_agent], list(squ_epi_len_agent))
            x_agent = pad_sequence(x_agent, padding_value=torch.tensor(0.0), batch_first=True).to(device) # n_batch x max_squ_epi_length_agent x dim_obs (+ dim_action)

            if h is not None:
                h_agent = h[agent_idx]
            else:
                h_agent = None
            
            gru_outputs_agent, h_next_agent = self.encoders[agent_idx](x_agent, h_agent)

            j_outputs_agent = self.recover_joint_feature(gru_outputs_agent, v_agent) * j_v.unsqueeze(-1) # n_batch x trace_len x dim_feature

            feature.append(j_outputs_agent.unsqueeze(0))
            if h_next_agent is not None:
                h_next.append(h_next_agent) 

        feature = torch.concat(feature, dim=0) # n_agent x n_batch x trace_len x dim_feature
        feature = feature.permute(1, 2, 0, 3).reshape(n_batch, trace_len, -1).contiguous() # n_batch x trace_len x (n_agent * dim_feature)
        if len(h_next):
            h_next = torch.concat(h_next, dim=0) # n_agent x n_layers x dim_h
            n_layers = h_next.size()[1]
            h_next = h_next.permute(1, 0, 2).reshape(n_layers, -1).contiguous() # n_layers x (n_agent * dim_h)
        else:
            h_next = None

        squ_j_epi_len = j_v.sum(1)
        max_j_epi_len = torch.max(squ_j_epi_len)
        feature = feature[j_v]

        if self.use_attention:
            joint_transformer_outputs = self.centralized_transformer(
                inputs_embeds=feature.reshape(feature.size()[0], n_agent, -1).contiguous(),
                attention_mask=torch.ones(feature.size()[0], n_agent).to(device),
                use_position_embedding=True
            )
            joint_transformer_outputs = joint_transformer_outputs['last_hidden_state'] # n_total_j_mac_steps x n_agent x dim_feature
            if self.value_head == "concat":
                state_value = self.value(joint_transformer_outputs.reshape((joint_transformer_outputs.size()[0], -1)))
            elif self.value_head == "share_linear_add":
                state_value = self.value(joint_transformer_outputs).sum(dim=1)
            elif self.value_head == "add_linear":
                state_value = self.value(joint_transformer_outputs.sum(dim=1))
            elif self.value_head == "mean_linear":
                state_value = self.value(joint_transformer_outputs.mean(dim=1))
            elif self.value_head == "max_linear":
                state_value = self.value(joint_transformer_outputs.max(dim=1)[0])
            elif self.value_head == "min_linear":
                state_value = self.value(joint_transformer_outputs.min(dim=1)[0])
            else:
                raise NotImplementedError

        else:
            feature = F.leaky_relu(self.fc(feature))
            state_value = self.value(feature)

        state_value = torch.split_with_sizes(state_value, list(squ_j_epi_len))
        state_value = pad_sequence(state_value, padding_value=torch.tensor(0.0), batch_first=True).reshape(n_batch, -1, 1).to(device) # n_batch x trace_len x 1

        return state_value, h_next
    


