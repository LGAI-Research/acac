import copy
from itertools import chain
import numpy as np
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from acac.acac_marl.cores.acac.models import AgentCentricGRUCritic

class Learner_ACAC_Micro_GAE(object):
    
    def __init__(self, 
                 env, 
                 controller, 
                 memory, 
                 gamma, 
                 obs_last_action=False,
                 a_lr=1e-2, 
                 c_lr=1e-2, 
                 c_mlp_layer_size=64, 
                 c_rnn_layer_size=64,
                 c_train_iteration=1, 
                 c_target_update_freq=50, 
                 n_train_repeat=1, 
                 n_minibatch=8,
                 tau=0.01,
                 grad_clip_value=None, 
                 grad_clip_norm=None,
                 n_step_TD=0, 
                 TD_lambda=0.0,
                 GAE_lambda=0.95,
                 device='cpu',
                 clip_ratio=0.1, 
                 vf_coef=0.5,
                 **kwargs):

        self.env = env
        self.n_agent = env.n_agent
        self.controller = controller
        self.memory = memory
        self.gamma = gamma

        self.a_lr = a_lr
        self.c_lr = c_lr
        self.c_mlp_layer_size = c_mlp_layer_size
        self.c_rnn_layer_size = c_rnn_layer_size
        self.c_train_iteration = c_train_iteration
        self.c_target_update_freq = c_target_update_freq
        self.n_train_repeat = n_train_repeat
        self.n_minibatch = n_minibatch

        self.obs_last_action = obs_last_action
        self.tau = tau
        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm
        self.n_step_TD = n_step_TD
        self.TD_lambda = TD_lambda
        self.GAE_lambda = GAE_lambda
        self.device = device
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef if self.controller.share_encoder else 1.0

        self._create_joint_critic()

        self.diagnostics = {}
        self.diagnostics[f'Joint/Value'] = []
        self.diagnostics[f'Joint/Advantage'] = []
        self.diagnostics[f'Joint/CriticLoss'] = []
        self.diagnostics[f'Joint/VfCoef'] = []
        for idx in range(self.n_agent):
            self.diagnostics[f'Agent{idx}/ActorLoss'] = []
            self.diagnostics[f'Agent{idx}/Entropy'] = []
            self.diagnostics[f'Agent{idx}/ISRatio'] = []
            self.diagnostics[f'Agent{idx}/ClipRate'] = []
        self._set_optimizer()

        print('Agent Centric Actor Critic with Micro-level GAE ====================================================')
        if self.controller.share_encoder:
            print('    with Shared Encoder')
        print('      with vf_coef = ', self.vf_coef)    
        if self.controller.use_popart:
            print('    with PopArt')
        

    def train(self, eps, c_hys_value, adv_hys_value, etrpy_w, critic_hys=False, adv_hys=False):

        batch, trace_len, epi_len = self.memory.sample()
        batch_size = len(batch)
        
        ### Make cen_batches and dec_batches (prepare training)    
        cen_batch, dec_batches = self._sep_joint_exps(batch)    
        cen_batch, cen_trace_len, cen_epi_len = self._squeeze_cen_exp(cen_batch, 
                                                                      batch_size, 
                                                                      trace_len)
        jobs, jr, n_jobs, j_terminate, mac_v_b, j_mac_v_b, j_discount, exp_valid, mac_st, jobs_seq, n_jobs_seq, j_gae_lambda = cen_batch

        dec_batches, dec_trace_lens, dec_epi_lens = self._squeeze_dec_exp(dec_batches, 
                                                                        batch_size, 
                                                                        trace_len, 
                                                                        mac_v_b)
        
        ### Prepare training
        with torch.no_grad():
            # Compute Gt, advantage
            old_init_value, old_bootstrap = self._get_bootstrap(jobs, n_jobs, j_mac_v_b, mac_v_b, n_jobs_seq)
            adv_value, Gt = self._get_gae_old(jr, old_init_value, old_bootstrap, j_discount, j_gae_lambda, j_terminate, cen_epi_len)
            
            ### Compute log_old_pi(a|s) for each agent
            old_log_pi_a_agent = []
            for agent, d_batch, d_trace_len, d_epi_len in zip(self.controller.agents, 
                                                        dec_batches, 
                                                        dec_trace_lens, 
                                                        dec_epi_lens):
                obs_agent, action_agent, discount_agent, exp_valid_agent, obs_mask_agent, obs_seq_agent, mac_st_j = d_batch
                
                if obs_agent.shape[1] == 0:
                    continue
                if not self.controller.time_emb:
                    obs_seq_agent = None

                action_logits_agent = agent.actor_net(obs_agent, eps=eps, time_emb=obs_seq_agent)[0]
                ### Append detached log_pi_a for each agent
                old_log_pi_a_agent.append(action_logits_agent.gather(-1, action_agent))

        ### Repeat training for n_train_repeat times with the same batch
        for _ in range(self.n_train_repeat):
            actor_critic_loss = 0

            ### Train joint critic
            values = self.joint_critic_net(jobs, mac_st, time_emb=jobs_seq)[0]
        
            V_value = torch.split_with_sizes(values[torch.amax(mac_st, dim=-1).to(torch.bool)], list(cen_epi_len))
            V_value = pad_sequence(V_value, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            TD = Gt - V_value

            if critic_hys:
                TD = torch.max(TD*c_hys_value, TD)
            joint_critic_loss = torch.sum(exp_valid * TD * TD) / exp_valid.sum()
            actor_critic_loss += self.vf_coef * joint_critic_loss
            self.diagnostics[f'Joint/CriticLoss'].append(joint_critic_loss.detach().cpu().numpy())
            self.diagnostics[f'Joint/Value'].append((torch.sum(exp_valid * V_value)/ exp_valid.sum()).detach().cpu().numpy())
            self.diagnostics[f'Joint/Advantage'].append((torch.sum(exp_valid * TD)/ exp_valid.sum()).detach().cpu().numpy())
            self.diagnostics[f'Joint/VfCoef'].append(self.vf_coef)

            if not self.controller.share_encoder:
                self.joint_critic_optimizer.zero_grad()
                joint_critic_loss.backward()
                if self.grad_clip_value:
                    clip_grad_value_(self.joint_critic_net.parameters(), self.grad_clip_value)
                if self.grad_clip_norm:
                    clip_grad_norm_(self.joint_critic_net.parameters(), self.grad_clip_norm)
                self.joint_critic_optimizer.step()

            ### Train joint critic
            for agent, d_batch, d_trace_len, epi_len, old_log_pi_a in zip(self.controller.agents, 
                                                        dec_batches, 
                                                        dec_trace_lens, 
                                                        dec_epi_lens,
                                                        old_log_pi_a_agent):
                
                obs_agent, action_agent, discount_agent, exp_valid_agent, obs_mask_agent, obs_seq_agent, mac_st_j = d_batch
                
                ### Get log_pi_a for each agent
                action_logits_agent = agent.actor_net(obs_agent, eps=eps, time_emb=obs_seq_agent)[0]
                log_pi_a = action_logits_agent.gather(-1, action_agent)

                ### Calculate entropy
                pi_entropy = torch.distributions.Categorical(logits=action_logits_agent * exp_valid_agent).entropy()
                pi_entropy = pi_entropy.view(obs_agent.shape[0], d_trace_len, 1) # (batch, d_trace_len, 1)

                ### Calculate actor loss
                is_ratio = torch.exp((log_pi_a - old_log_pi_a) * exp_valid_agent) 
                clipped_rate = torch.sum(torch.logical_or(is_ratio < 1-self.clip_ratio, is_ratio > 1 + self.clip_ratio).to(float)) / torch.sum(exp_valid_agent)

                adv_agent = self._squeeze_tensor_by_mac_valid(adv_value, mac_st_j)

                pg_loss1 = exp_valid_agent * adv_agent * is_ratio
                pg_loss2 = exp_valid_agent * adv_agent * torch.clamp(is_ratio, min=1-self.clip_ratio, max=1+self.clip_ratio)
                agent.actor_loss = (-torch.sum(torch.min(pg_loss1, pg_loss2), dim=-1, keepdim=True) * exp_valid_agent).sum() / exp_valid_agent.sum()
                actor_critic_loss += agent.actor_loss

                self.diagnostics[f'Agent{agent.idx}/ActorLoss'].append(agent.actor_loss.detach().cpu().numpy())
                self.diagnostics[f'Agent{agent.idx}/Entropy'].append((torch.sum(exp_valid_agent * pi_entropy) / exp_valid_agent.sum()).detach().cpu().numpy())
                self.diagnostics[f'Agent{agent.idx}/ISRatio'].append((torch.sum(is_ratio * exp_valid_agent) / torch.sum(exp_valid_agent)).detach().cpu().numpy())
                self.diagnostics[f'Agent{agent.idx}/ClipRate'].append(clipped_rate.detach().cpu().numpy())

                if not self.controller.share_encoder:
                    agent.actor_optimizer.zero_grad()
                    agent.actor_loss.backward()
                    if self.grad_clip_value:
                        clip_grad_value_(agent.actor_net.parameters(), self.grad_clip_value)
                    if self.grad_clip_norm:
                        clip_grad_norm_(agent.actor_net.parameters(), self.grad_clip_norm)
                    agent.actor_optimizer.step()

            if self.controller.share_encoder:
                self.actor_critic_optimizer.zero_grad()
                actor_critic_loss.backward()
                if self.grad_clip_value:
                    clip_grad_value_(self.actor_critic_parameters, self.grad_clip_value)
                if self.grad_clip_norm:
                    clip_grad_norm_(self.actor_critic_parameters, self.grad_clip_norm)
                self.actor_critic_optimizer.step()

    def update_critic_target_net(self, soft=False):
        if not soft:
            self.joint_critic_tgt_net.load_state_dict(self.joint_critic_net.state_dict())
        else:
            with torch.no_grad():
                for q, q_targ in zip(self.joint_critic_net.parameters(), self.joint_critic_tgt_net.parameters()):
                    q_targ.data.mul_(1 - self.tau)
                    q_targ.data.add_(self.tau * q.data)

    def update_actor_target_net(self, soft=False):
        for agent in self.controller.agents:
            if not soft:
                agent.actor_tgt_net.load_state_dict(agent.actor_net.state_dict())
            else:
                with torch.no_grad():
                    for q, q_targ in zip(agent.actor_net.parameters(), agent.actor_tgt_net.parameters()):
                        q_targ.data.mul_(1 - self.tau)
                        q_targ.data.add_(self.tau * q.data)

    def get_diagnostics(self):
        diag = copy.deepcopy(self.diagnostics)
        for k in self.diagnostics.keys():
            self.diagnostics[k] = []
        return diag

    def _create_joint_critic(self):
        input_dim = self._get_input_shape()
        
        if self.controller.share_encoder:
            encoders = [agent.encoder for agent in self.controller.agents]
            encoders_tgt = [agent.encoder_tgt for agent in self.controller.agents]
        else:
            encoders = encoders_tgt = None
        self.joint_critic_net = AgentCentricGRUCritic(
                                    input_dim,
                                    1,
                                    self.c_mlp_layer_size,
                                    self.c_rnn_layer_size,
                                    n_agent=self.n_agent,
                                    encoders=encoders,
                                    use_attention=self.controller.use_attention,
                                    cc_n_head=self.controller.cc_n_head,
                                    value_head=self.controller.value_head,
                                    use_time_emb=self.controller.time_emb,
                                    time_emb_alg = self.controller.time_emb_alg,
                                    time_emb_dim=self.controller.time_emb_dim, 
                                    max_timestep=self.controller.max_timestep,
                                    use_popart=self.controller.use_popart,
                                    duplicate=self.controller.duplicate,
                                    ).to(self.device)
        self.joint_critic_tgt_net = AgentCentricGRUCritic(
                                    input_dim,
                                    1,
                                    self.c_mlp_layer_size,
                                    self.c_rnn_layer_size,
                                    n_agent=self.n_agent,
                                    encoders=encoders_tgt,
                                    use_attention=self.controller.use_attention,
                                    cc_n_head=self.controller.cc_n_head,
                                    value_head=self.controller.value_head,
                                    use_time_emb=self.controller.time_emb,
                                    time_emb_alg = self.controller.time_emb_alg,
                                    time_emb_dim=self.controller.time_emb_dim, 
                                    max_timestep=self.controller.max_timestep,
                                    use_popart=self.controller.use_popart,
                                    duplicate=self.controller.duplicate,
                                    ).to(self.device)
        self.joint_critic_tgt_net.load_state_dict(self.joint_critic_net.state_dict())

    def _get_input_shape(self):
        if not self.obs_last_action:
            return self.env.obs_size # sum(self.env.obs_size)
        else:
            return [o_dim + a_dim for o_dim, a_dim in zip(*[self.env.obs_size, self.env.n_action])]

    def _set_optimizer(self):
        self.actor_critic_parameters = set(self.joint_critic_net.parameters())
        for agent in self.controller.agents:
            self.actor_critic_parameters = self.actor_critic_parameters | set(agent.actor_net.parameters())
        
        if self.controller.share_encoder:
            self.actor_critic_optimizer = Adam(self.actor_critic_parameters, lr=self.a_lr)
            self.joint_critic_optimizer = None
        else:  
            for agent in self.controller.agents:
                agent.actor_optimizer = Adam(agent.actor_net.parameters(), lr=self.a_lr)
            self.joint_critic_optimizer = Adam(self.joint_critic_net.parameters(), lr=self.c_lr)
            self.actor_critic_optimizer = None

    def _squeeze_tensor_by_mac_valid(self, tensor, mac_v, padding_value=0.0, popart=False):
        """
        tensor: (batch, trace_len, n_agent) or (batch, trace_len) 
        mac_v: (batch, trace_len) 
        """
        squ_epi_len = mac_v.sum(1)
        squ_tensor = torch.split_with_sizes(tensor[mac_v], list(squ_epi_len))   
        
        if popart:
            squ_tensor = [self.joint_critic_net.value.denormalize(squ_t) for squ_t in squ_tensor] 

        padded_tensor = pad_sequence(squ_tensor, padding_value=torch.tensor(padding_value), batch_first=True).to(self.device)
        return padded_tensor

    def _get_bootstrap(self, squ_jo_b, squ_n_jo_b, squ_j_mac_v_b, squ_mac_v_b, squ_n_jo_seq):
        jobs = torch.cat([squ_jo_b[:,0].unsqueeze(1),squ_n_jo_b],dim=1)
        mac_v = torch.cat([torch.ones([squ_mac_v_b.shape[0], 1, squ_mac_v_b.shape[2]]).to(self.device), squ_mac_v_b],dim=1).to(torch.bool)
        n_jo_seq = torch.cat([torch.ones([squ_n_jo_seq.shape[0], 1]).to(self.device), squ_n_jo_seq], dim=1).to(torch.int64)

        bootstrap_values = self.joint_critic_tgt_net(
                jobs,
                mac_v,
                time_emb=n_jo_seq 
                )[0]
        init_values = bootstrap_values[:,0,:].unsqueeze(1)
        if self.controller.use_popart:
            init_values = self.joint_critic_net.value.denormalize(init_values)
        squ_bootstrap = self._squeeze_tensor_by_mac_valid(bootstrap_values[:,1:,:], squ_j_mac_v_b, popart=self.controller.use_popart)

        return init_values, squ_bootstrap


    def _sep_joint_exps(self, joint_exps):
        cen_exps = []
        dec_exps = [[] for _ in range(self.n_agent)]
        for o, a_st, _avail_a, a, r, j_r, n_o, _n_avail_a, t, mac_v, j_mac_v, exp_v in chain(*joint_exps):
            cen_exps.append([torch.cat(o, dim=1).view(1,-1), 
                         torch.cat(a_st, dim=1).view(1,-1), # Modified: a_st_all (n_batch * trace_len, n_agent) 
                         max(a_st),
                         j_r, 
                         torch.cat(n_o, dim=1).view(1,-1), 
                         t, 
                         torch.cat(mac_v).view(1,-1),
                         j_mac_v,
                         exp_v[0]])
            
            for i in range(self.n_agent):
                dec_exps[i].append([o[i], 
                                a_st[i],
                                max(a_st),
                                a[i], 
                                r[i], 
                                n_o[i], 
                                t, 
                                mac_v[i], 
                                j_mac_v,
                                exp_v[i]])
        return cen_exps, dec_exps

    def _squeeze_dec_exp(self, dec_batches, batch_size, trace_len, j_padded_mac_v_b):

        """
        squeeze experience for each agent and re-padding
        """

        squ_dec_batches = []
        squ_epi_lens = []
        squ_trace_lens = []

        j_padded_mac_v_b = j_padded_mac_v_b.to(self.device)

        for idx, batch in enumerate(dec_batches):
            # seperate elements in the batch
            obs_b, action_start_b, jaction_start_b, action_b, reward_b, next_obs_b, terminate_b, mac_valid_b, j_mac_valid_b, exp_valid_b = zip(*batch)
            assert len(obs_b) == trace_len * batch_size, "number of states mismatch ..."
            assert len(next_obs_b) == trace_len * batch_size, "number of next states mismatch ..."
            o_b = torch.cat(obs_b).view(batch_size, trace_len, -1).to(self.device)
            a_b = torch.cat(action_b).view(batch_size, trace_len, -1).to(self.device)
            a_st_b = torch.cat(action_start_b).view(batch_size, trace_len).to(self.device)
            ja_st_b = torch.cat(jaction_start_b).view(batch_size, trace_len).to(self.device)
            mac_v_b = torch.cat(mac_valid_b).view(batch_size, trace_len).to(self.device)
            j_mac_v_b = torch.cat(j_mac_valid_b).view(batch_size, trace_len).to(self.device)
            exp_v_b = torch.cat(exp_valid_b).view(batch_size, trace_len, -1).to(self.device)
            discount_b = torch.pow(torch.ones(o_b.shape[0],1)*self.gamma, torch.arange(o_b.shape[1])).unsqueeze(-1).to(self.device) 


            if not (ja_st_b.sum(1) == j_mac_v_b.sum(1)).all():
                self._mac_start_filter(ja_st_b, j_mac_v_b)
            assert all(ja_st_b.sum(1) == j_mac_v_b.sum(1)), "mask for joint mac start does not match with mask of joint mac done ..."

            if not (a_st_b.sum(1) == mac_v_b.sum(1)).all():
                self._mac_start_filter(a_st_b, mac_v_b)
            assert all(a_st_b.sum(1) == mac_v_b.sum(1)), "mask for mac start does not match with mask of mac done ..."

            # squeeze process
            squ_epi_len = mac_v_b.sum(1)
            assert all(squ_epi_len == j_padded_mac_v_b[:,:,idx].sum(1)), "Valid mask doesn't match ..."
            
            squ_o_b = self._squeeze_tensor_by_mac_valid(o_b, mac_v_b)
            squ_a_b = self._squeeze_tensor_by_mac_valid(a_b, mac_v_b)
            squ_discount_b = self._squeeze_tensor_by_mac_valid(discount_b, mac_v_b)
            squ_exp_v_b = self._squeeze_tensor_by_mac_valid(exp_v_b, mac_v_b)
            
            squ_o_b_attn_mask = self._generate_masking(squ_epi_len).to(self.device)
            
            if self.controller.time_emb:
                o_seq = torch.stack([torch.arange(1,o_b.shape[1]+1) for _ in range(0,o_b.shape[0])],dim=0).to(self.device)
                squ_o_seq = torch.split_with_sizes(o_seq[a_st_b], list(squ_epi_len)) 
                squ_o_seq = pad_sequence(squ_o_seq, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            else:
                squ_o_seq = None

            squ_j_epi_len = j_mac_v_b.sum(1)
            squ_mac_st_jb = torch.split_with_sizes(a_st_b[ja_st_b], list(squ_j_epi_len))
            squ_mac_st_jb = pad_sequence(squ_mac_st_jb, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)

            squ_dec_batches.append((squ_o_b,
                                    squ_a_b,
                                    squ_discount_b,
                                    squ_exp_v_b,
                                    squ_o_b_attn_mask,
                                    squ_o_seq,
                                    squ_mac_st_jb))

            squ_epi_lens.append(squ_epi_len)
            squ_trace_lens.append(squ_o_b.shape[1])

        return squ_dec_batches, squ_trace_lens, squ_epi_lens

    def _squeeze_cen_exp(self, cen_batch, batch_size, trace_len):

        """
        squeeze experience for each agent and re-padding
        """

        # seperate elements in the batch
        jobs_b, action_start_b, jaction_start_b, reward_b, next_jobs_b, terminate_b, mac_valid_b, j_mac_valid_b, exp_valid_b = zip(*cen_batch)
        assert len(jobs_b) == trace_len * batch_size, "number of states mismatch ..."
        assert len(next_jobs_b) == trace_len * batch_size, "number of next states mismatch ..."
        jo_b = torch.cat(jobs_b).view(batch_size, trace_len, -1).to(self.device)

        a_st_b = torch.cat(action_start_b).view(batch_size, trace_len, -1).to(self.device) 
        ja_st_b = torch.cat(jaction_start_b).view(batch_size, trace_len).to(self.device)

        r_b = torch.cat(reward_b).view(batch_size, trace_len, -1).to(self.device)
        n_jo_b = torch.cat(next_jobs_b).view(batch_size, trace_len, -1).to(self.device)
        t_b = torch.cat(terminate_b).view(batch_size, trace_len, -1).to(self.device)

        mac_v_b = torch.cat(mac_valid_b).view(batch_size, trace_len, -1).to(self.device)
        j_mac_v_b = torch.cat(j_mac_valid_b).view(batch_size, trace_len).to(self.device)
        exp_v_b = torch.cat(exp_valid_b).view(batch_size, trace_len, -1).to(self.device)
        discount_b = torch.pow(torch.ones(jo_b.shape[0],1)*self.gamma, torch.arange(jo_b.shape[1])).unsqueeze(-1).to(self.device)
        gae_lambda_b = torch.pow(torch.ones(jo_b.shape[0],1)*self.GAE_lambda, torch.arange(jo_b.shape[1])).unsqueeze(-1).to(self.device)

        if not (ja_st_b.sum(1) == j_mac_v_b.sum(1)).all():
            self._mac_start_filter(ja_st_b, j_mac_v_b)
        assert all(ja_st_b.sum(1) == j_mac_v_b.sum(1)), "mask for joint mac start does not match with mask of joint mac done ..."

        squ_epi_len = j_mac_v_b.sum(1)
        squ_jo_b = self._squeeze_tensor_by_mac_valid(jo_b, j_mac_v_b) # joint observation
        squ_r_b = self._squeeze_tensor_by_mac_valid(r_b, j_mac_v_b) # reward
        squ_n_jo_b = self._squeeze_tensor_by_mac_valid(n_jo_b, j_mac_v_b) # next joint observation
        squ_t_b = self._squeeze_tensor_by_mac_valid(t_b, j_mac_v_b, padding_value=1.0) # terminated
        squ_mac_st_b = self._squeeze_tensor_by_mac_valid(a_st_b, ja_st_b) # macro action start
        squ_mac_v_b = self._squeeze_tensor_by_mac_valid(mac_v_b, j_mac_v_b) # macto action valid
        squ_j_mac_v_b = self._squeeze_tensor_by_mac_valid(j_mac_v_b, j_mac_v_b) # joint macro action valid
        squ_exp_v_b = self._squeeze_tensor_by_mac_valid(exp_v_b, j_mac_v_b) # experiment valid
        squ_discount_b = self._squeeze_tensor_by_mac_valid(discount_b, j_mac_v_b) # discounts
        squ_gae_lambda_b = self._squeeze_tensor_by_mac_valid(gae_lambda_b, j_mac_v_b) # discounts

        if self.controller.time_emb:
            jo_seq = torch.stack([torch.arange(1,jo_b.shape[1]+1) for _ in range(0,jo_b.shape[0])],dim=0).to(self.device)
            n_jo_seq = torch.stack([torch.arange(2,jo_b.shape[1]+2) for _ in range(0,jo_b.shape[0])],dim=0).to(self.device)
            squ_jo_seq = self._squeeze_tensor_by_mac_valid(jo_seq, ja_st_b)
            squ_n_jo_seq = self._squeeze_tensor_by_mac_valid(n_jo_seq, j_mac_v_b)
        else:
            squ_jo_seq = squ_n_jo_seq = None

        squ_cen_batch = (squ_jo_b,
                         squ_r_b,
                         squ_n_jo_b,
                         squ_t_b,
                         squ_mac_v_b,
                         squ_j_mac_v_b,
                         squ_discount_b,
                         squ_exp_v_b,
                         squ_mac_st_b,
                         squ_jo_seq,
                         squ_n_jo_seq,
                         squ_gae_lambda_b,
                         )
        return squ_cen_batch, squ_jo_b.shape[1], squ_epi_len

    def _generate_masking(self, epi_length):
        max_seq_len = torch.max(epi_length).item()
        masking = []
        for length in epi_length:
            seq_len = length.item()
            pad_len = max_seq_len-seq_len
            masking.append(torch.cat([torch.ones([1,seq_len]),torch.zeros([1,pad_len])],dim=1)) #1,max_seq_len
        
        return torch.cat(masking,dim=0)

    def _mac_start_filter(self, mac_start, mac_end):

        mask = mac_start.sum(1) != mac_end.sum(1)
        selected_items = mac_start[mask]
        indices = torch.cat([i[-1].view(-1,2) for i in torch.split_with_sizes(selected_items.nonzero(as_tuple=False), 
                                                                              list(selected_items.sum(1)))], 
                                                                              dim=0)
        selected_items.scatter_(-1, indices[:,1].view(-1,1), 0.0)
        mac_start[mask] = selected_items
    
    def _get_gae_old(self, reward, initial_value, bootstrap, discount, gae_lambda, terminate, epi_len):
        # reward: n_batch x max_epi_length_agent
        mac_discount = discount / torch.cat((self.gamma**-1*torch.ones((discount.shape[0],1,1)).to(self.device),
                                             discount[:,0:-1,:]),
                                             axis=1) 
        mac_lambda = gae_lambda / torch.cat((self.GAE_lambda**-1*torch.ones((discount.shape[0],1,1)).to(self.device),
                                             gae_lambda[:,0:-1,:]),
                                             axis=1) 
        
        mask = mac_discount.isnan()
        mac_discount[mask] = 0.0
        advantage = torch.zeros_like(reward).to(self.device)
        Gt = torch.zeros_like(reward).to(self.device)
        Gt_valid = torch.zeros_like(reward).to(torch.bool).to(self.device)

        for epi_idx, epi_r in enumerate(reward):
            end_step_idx = epi_len[epi_idx]-1

            if not terminate[epi_idx][end_step_idx]:
                advantage[epi_idx][end_step_idx] = epi_r[end_step_idx] + mac_discount[epi_idx][end_step_idx] * bootstrap[epi_idx][end_step_idx] - bootstrap[epi_idx][end_step_idx-1]
            else:
                advantage[epi_idx][end_step_idx] = epi_r[end_step_idx] - bootstrap[epi_idx][end_step_idx-1]
            
            for idx in range(end_step_idx-1, -1, -1):
                if idx == 0:
                    delta = epi_r[idx] + mac_discount[epi_idx][idx] * bootstrap[epi_idx][idx] - initial_value[epi_idx]
                else:
                    delta = epi_r[idx] + mac_discount[epi_idx][idx] * bootstrap[epi_idx][idx] - bootstrap[epi_idx][idx-1]
                advantage[epi_idx][idx] = delta + mac_discount[epi_idx][idx] * mac_lambda[epi_idx][idx] * advantage[epi_idx][idx + 1]
            value = torch.zeros_like(reward[epi_idx])
            value[:end_step_idx] = torch.cat([initial_value[epi_idx], bootstrap[epi_idx][:end_step_idx-1]], dim=0)
            Gt[epi_idx] = advantage[epi_idx] + value
            Gt_valid[epi_idx][:epi_len[epi_idx]] = True

        if self.controller.use_popart:
            Gt_ = Gt[Gt_valid].unsqueeze(1)
            self.joint_critic_net.value.update(Gt_)
            Gt_ = self.joint_critic_net.value.normalize(Gt_)
            Gt_ = torch.split_with_sizes(Gt_, list(epi_len))
            Gt = pad_sequence(Gt_, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        
        return advantage, Gt