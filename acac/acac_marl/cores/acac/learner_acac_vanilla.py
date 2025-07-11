import copy
from itertools import chain
import numpy as np
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from acac.acac_marl.cores.acac.models import AgentCentricGRUCritic

class Learner_ACAC_Vanilla(object):
    
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
                 clip_ratio=0.1):

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

        self._create_joint_critic()

        self.diagnostics = {}
        self.diagnostics[f'Joint/Value'] = []
        self.diagnostics[f'Joint/Advantage'] = []
        self.diagnostics[f'Joint/CriticLoss'] = []
        for idx in range(self.n_agent):
            self.diagnostics[f'Agent{idx}/ActorLoss'] = []
            self.diagnostics[f'Agent{idx}/Entropy'] = []
        self._set_optimizer()

        print('Agent Centric Actor Critic (Vanilla AC) ====================================================')

    def train(self, eps, c_hys_value, adv_hys_value, etrpy_w, critic_hys=False, adv_hys=False):

        batch, trace_len, epi_len = self.memory.sample()
        batch_size = len(batch)

        ############################# Squeeze batches for both cent, decent ###################################
        # Squeeze cent
        cen_batch = self._cat_joint_exps(batch)
        cen_batch, cen_trace_len, cen_epi_len = self._squeeze_cen_exp(cen_batch, 
                                                                      batch_size, 
                                                                      trace_len)

        jobs, reward, n_jobs, terminate, mac_v_b, discount, exp_valid, bootstrap, mac_st, jobs_seq = cen_batch

        if not self.controller.time_emb:
            jobs_seq = None

        if jobs.shape[1] == 0:
            return

        if not self.TD_lambda:
            Gt = self._get_bootstrap_return(reward, 
                                            bootstrap,
                                            discount,
                                            terminate, 
                                            cen_epi_len)
        else:
            Gt = self._get_td_lambda_return(jobs.shape[0], 
                                            cen_trace_len, 
                                            cen_epi_len, 
                                            reward, 
                                            bootstrap,
                                            terminate)

        ##############################  calculate critic loss and optimize the critic_net ####################################
        for _ in range(self.n_train_repeat):
            for _ in range(self.c_train_iteration):
                values = self.joint_critic_net(jobs, mac_st, time_emb=jobs_seq)[0]            
                V_value = torch.split_with_sizes(values[torch.amax(mac_st, dim=-1).to(torch.bool)], list(cen_epi_len))
                V_value = pad_sequence(V_value, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
                TD = Gt - V_value

                if critic_hys:
                    TD = torch.max(TD*c_hys_value, TD)
                joint_critic_loss = torch.sum(exp_valid * TD * TD) / exp_valid.sum()
                self.diagnostics[f'Joint/CriticLoss'].append(joint_critic_loss.detach().cpu().numpy())
                self.diagnostics[f'Joint/Value'].append((torch.sum(exp_valid * V_value)/ exp_valid.sum()).detach().cpu().numpy())
                self.diagnostics[f'Joint/Advantage'].append((torch.sum(exp_valid * TD)/ exp_valid.sum()).detach().cpu().numpy())
                
                self.joint_critic_optimizer.zero_grad()
                joint_critic_loss.backward()
                if self.grad_clip_value:
                    clip_grad_value_(self.joint_critic_net.parameters(), self.grad_clip_value)
                if self.grad_clip_norm:
                    clip_grad_norm_(self.joint_critic_net.parameters(), self.grad_clip_norm)
                self.joint_critic_optimizer.step()

            ##############################  calculate adv using updated critic ####################################

            V_value = self.joint_critic_net(jobs, mac_st, time_emb=jobs_seq)[0].detach()


            ##############################  calculate actor loss and optimize actors ####################################

            # Squeeze decent
            dec_batches = self._sep_joint_exps(batch)
            dec_batches, dec_trace_lens, dec_epi_lens = self._squeeze_dec_exp(dec_batches, 
                                                                            batch_size, 
                                                                            trace_len, 
                                                                            mac_v_b)

            for agent, batch, trace_len, epi_len in zip(self.controller.agents, 
                                                        dec_batches, 
                                                        dec_trace_lens, 
                                                        dec_epi_lens):

                obs, action, discount, exp_valid, obs_mask, obs_seq, mac_st_j = batch

                if obs.shape[1] == 0:
                    continue
                if not self.controller.time_emb:
                    obs_seq = None
                Gt_agent = torch.split_with_sizes(Gt[mac_st_j], list(epi_len))
                V_value_agent = torch.split_with_sizes(V_value[mac_st_j], list(epi_len))

                Gt_agent = pad_sequence(Gt_agent, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
                V_value_agent = pad_sequence(V_value_agent, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)

                adv = Gt_agent - V_value_agent

                action_logits = agent.actor_net(obs, eps=eps, time_emb=obs_seq)[0]
                # log_pi(a|s) 
                log_pi_a = action_logits.gather(-1, action)
                # H(pi(.|s)) used as exploration bonus
                pi_entropy = torch.distributions.Categorical(logits=action_logits).entropy().view(obs.shape[0], 
                                                                                                  trace_len, 
                                                                                                  1)
                # actor loss
                actor_loss = torch.sum(exp_valid * discount * (log_pi_a * adv + etrpy_w * pi_entropy), dim=1)
                agent.actor_loss = -1 * torch.sum(actor_loss) / exp_valid.sum()

                self.diagnostics[f'Agent{agent.idx}/ActorLoss'].append(agent.actor_loss.detach().cpu().numpy())
                self.diagnostics[f'Agent{agent.idx}/Entropy'].append((torch.sum(exp_valid * pi_entropy) / exp_valid.sum()).detach().cpu().numpy())

                ############################# optimize each actor-net ########################################
                agent.actor_optimizer.zero_grad()
                agent.actor_loss.backward()
                if self.grad_clip_value:
                    clip_grad_value_(agent.actor_net.parameters(), self.grad_clip_value)
                if self.grad_clip_norm:
                    clip_grad_norm_(agent.actor_net.parameters(), self.grad_clip_norm)
                agent.actor_optimizer.step()

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
                                    duplicate=self.controller.duplicate,
                                    ).to(self.device)
        self.joint_critic_tgt_net.load_state_dict(self.joint_critic_net.state_dict())

    def _get_input_shape(self):
        if not self.obs_last_action:
            return self.env.obs_size # sum(self.env.obs_size)
        else:
            return [o_dim + a_dim for o_dim, a_dim in zip(*[self.env.obs_size, self.env.n_action])]

    def _set_optimizer(self):
        for agent in self.controller.agents:
            agent.actor_optimizer = Adam(agent.actor_net.parameters(), lr=self.a_lr)
        self.joint_critic_optimizer = Adam(self.joint_critic_net.parameters(), lr=self.c_lr)

    def _sep_joint_exps(self, joint_exps):

        """
        seperate the joint experience for individual agents
        """

        exps = [[] for _ in range(self.n_agent)]
        for o, a_st, avail_a, a, r, j_r, n_o, n_avail_a, t, mac_v, j_mac_v, exp_v in chain(*joint_exps):
            for i in range(self.n_agent):
                exps[i].append([o[i], 
                                a_st[i],
                                max(a_st),
                                a[i], 
                                r[i], 
                                n_o[i], 
                                t, 
                                mac_v[i], 
                                j_mac_v,
                                exp_v[i]])
        return exps

    def _cat_joint_exps(self, joint_exps):

        """
        seperate the joint experience for individual agents
        """

        exps = []
        for o, a_st, avail_a, a, r, j_r, n_o, n_avail_a, t, mac_v, j_mac_v, exp_v in chain(*joint_exps):
            exps.append([torch.cat(o, dim=1).view(1,-1), 
                         torch.cat(a_st, dim=1).view(1,-1), # Modified: a_st_all (n_batch * trace_len, n_agent) 
                         max(a_st),
                         j_r, 
                         torch.cat(n_o, dim=1).view(1,-1), 
                         t, 
                         torch.cat(mac_v).view(1,-1),
                         j_mac_v,
                         exp_v[0]])
        return exps

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
            squ_o_b_attn_mask = self._generate_masking(squ_epi_len).to(self.device)
            squ_o_b = torch.split_with_sizes(o_b[mac_v_b], list(squ_epi_len))
            squ_a_b = torch.split_with_sizes(a_b[mac_v_b], list(squ_epi_len))
            # squ_adv_b = torch.split_with_sizes(adv_value[j_padded_mac_v_b[:,:,idx]], list(squ_epi_len))
            squ_exp_v_b = torch.split_with_sizes(exp_v_b[mac_v_b], list(squ_epi_len))
            squ_discount_b = torch.split_with_sizes(discount_b[mac_v_b], list(squ_epi_len))

            # re-padding
            squ_o_b = pad_sequence(squ_o_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_a_b = pad_sequence(squ_a_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            # squ_adv_b = pad_sequence(squ_adv_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_exp_v_b = pad_sequence(squ_exp_v_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_discount_b = pad_sequence(squ_discount_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)

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

        if not (ja_st_b.sum(1) == j_mac_v_b.sum(1)).all():
            self._mac_start_filter(ja_st_b, j_mac_v_b)
        assert all(ja_st_b.sum(1) == j_mac_v_b.sum(1)), "mask for joint mac start does not match with mask of joint mac done ..."

        # squeeze process
        squ_epi_len = j_mac_v_b.sum(1)
        squ_jo_b = torch.split_with_sizes(jo_b[j_mac_v_b], list(squ_epi_len))
        squ_r_b = torch.split_with_sizes(r_b[j_mac_v_b], list(squ_epi_len))
        squ_n_jo_b = torch.split_with_sizes(n_jo_b[j_mac_v_b], list(squ_epi_len))
        squ_t_b = torch.split_with_sizes(t_b[j_mac_v_b], list(squ_epi_len))
        squ_mac_st_b = torch.split_with_sizes(a_st_b[ja_st_b], list(squ_epi_len))
        squ_mac_v_b = torch.split_with_sizes(mac_v_b[j_mac_v_b], list(squ_epi_len))
        squ_j_mac_v_b = torch.split_with_sizes(j_mac_v_b[j_mac_v_b], list(squ_epi_len))
        squ_exp_v_b = torch.split_with_sizes(exp_v_b[j_mac_v_b], list(squ_epi_len))
        squ_discount_b = torch.split_with_sizes(discount_b[j_mac_v_b], list(squ_epi_len))

        # re-padding
        squ_jo_b = pad_sequence(squ_jo_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_r_b = pad_sequence(squ_r_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_n_jo_b = pad_sequence(squ_n_jo_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_t_b = pad_sequence(squ_t_b, padding_value=torch.tensor(1.0), batch_first=True).to(self.device)
        squ_mac_st_b = pad_sequence(squ_mac_st_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_mac_v_b = pad_sequence(squ_mac_v_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_j_mac_v_b = pad_sequence(squ_j_mac_v_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_exp_v_b = pad_sequence(squ_exp_v_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_discount_b = pad_sequence(squ_discount_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)


        if self.controller.time_emb:
            jo_seq = torch.stack([torch.arange(1,jo_b.shape[1]+1) for _ in range(0,jo_b.shape[0])],dim=0).to(self.device)                
            n_jo_seq = torch.stack([torch.arange(1,jo_b.shape[1]+1) for _ in range(0,jo_b.shape[0])],dim=0).to(self.device)

            squ_jo_seq = torch.split_with_sizes(jo_seq[ja_st_b], list(squ_epi_len))
            squ_n_jo_seq = list(torch.split_with_sizes(n_jo_seq[j_mac_v_b], list(squ_epi_len)))
            
            for idx, seq in enumerate(squ_n_jo_seq) :
                # print(f'{idx}| seq: ', seq)
                squ_n_jo_seq[idx] = torch.cat([seq,(seq[-1]+1).unsqueeze(0).to(self.device)],dim=0)

            squ_jo_seq = pad_sequence(squ_jo_seq, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_n_jo_seq = pad_sequence(squ_n_jo_seq, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        else:
            squ_jo_seq = squ_n_jo_seq = None

        bootstrap_values = self.joint_critic_tgt_net(
                torch.cat([squ_jo_b[:,0].unsqueeze(1),squ_n_jo_b],dim=1),
                torch.cat([torch.ones([squ_mac_v_b.shape[0], 1, squ_mac_v_b.shape[2]]).to(self.device), squ_mac_v_b],dim=1).to(torch.bool),
                time_emb=squ_n_jo_seq 
                )[0].detach()[:,1:,:][squ_j_mac_v_b]

        squ_bootstraps = torch.split_with_sizes(bootstrap_values, list(squ_epi_len))
        squ_bootstraps = pad_sequence(squ_bootstraps, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)

        squ_cen_batch = (squ_jo_b,
                         squ_r_b,
                         squ_n_jo_b,
                         squ_t_b,
                         squ_mac_v_b,
                         squ_discount_b,
                         squ_exp_v_b,
                         squ_bootstraps,
                         squ_mac_st_b,
                         squ_jo_seq,
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

    def _get_bootstrap_return(self, reward, bootstrap, discount, terminate, epi_len):
        mac_discount = discount / torch.cat((self.gamma**-1*torch.ones((discount.shape[0],1,1)).to(self.device),
                                             discount[:,0:-1,:]),
                                             axis=1) 
        mask = mac_discount.isnan()
        mac_discount[mask] = 0.0
        if self.n_step_TD and self.n_step_TD != 1:
            # implement n-step bootstrap
            Gt = copy.deepcopy(reward)
            for epi_idx, epi_r in enumerate(Gt):
                end_step_idx = epi_len[epi_idx]-1
                if not terminate[epi_idx][end_step_idx]:
                    epi_r[end_step_idx] += mac_discount[epi_idx][end_step_idx] * bootstrap[epi_idx][end_step_idx]
                for idx in range(end_step_idx-1, -1, -1):
                    if idx > end_step_idx - self.n_step_TD:
                        epi_r[idx] = epi_r[idx] + mac_discount[epi_idx][idx] * epi_r[idx+1]
                    else:
                        if idx == 0:
                            epi_r[idx] = self._get_n_step_discounted_bootstrap_return(reward[epi_idx][idx:idx+self.n_step_TD], 
                                                                                      bootstrap[epi_idx][idx+self.n_step_TD-1],
                                                                                      discount[epi_idx][idx:idx+self.n_step_TD] / self.gamma**-1)
                        else:
                            epi_r[idx] = self._get_n_step_discounted_bootstrap_return(reward[epi_idx][idx:idx+self.n_step_TD], 
                                                                                      bootstrap[epi_idx][idx+self.n_step_TD-1],
                                                                                      discount[epi_idx][idx:idx+self.n_step_TD] / discount[epi_idx][idx-1])
        else:
            Gt = reward + mac_discount * bootstrap * (-terminate + 1)
        return Gt

    def _get_n_step_discounted_bootstrap_return(self, reward, bootstrap, discount):
        rewards = torch.cat((reward, bootstrap.reshape(-1,1)), axis=0)
        discounts = torch.cat((torch.ones((1,1)).to(self.device), discount), axis=0)
        Gt = torch.sum(discounts * rewards) 
        return Gt

    def _get_td_lambda_return(self, batch_size, trace_len, epi_len, reward, bootstrap, terminate):
        # calculate MC returns
        Gt = self._get_discounted_return(reward, bootstrap, terminate, epi_len)
        # calculate n-step bootstrap returns
        self.n_step_TD = 0
        n_step_part = self._get_bootstrap_return(reward, bootstrap, terminate, epi_len)
        for n in range(2, trace_len):
            self.n_step_TD=n
            next_n_step_part = self._get_bootstrap_return(reward, bootstrap, terminate, epi_len)
            n_step_part = torch.cat([n_step_part, next_n_step_part], dim=-1)
        # calculate the lmda for n-step bootstrap part
        lmdas = torch.pow(torch.ones(1,1)*self.TD_lambda, torch.arange(trace_len-1)).repeat(trace_len, 1).unsqueeze(0).repeat(batch_size,1,1)
        mask = (torch.arange(trace_len).view(-1,1) + torch.arange(trace_len-1).view(1,-1)).squeeze(0).repeat(batch_size,1,1)
        mask = mask >= epi_len.view(batch_size, -1, 1)-1
        lmdas[mask] = 0.0
        # calculate the lmda for MC part
        MC_lmdas = torch.zeros_like(Gt)
        for epi_id, length in enumerate(epi_len):
            last_step_lmda = torch.pow(torch.ones(1,1)*self.TD_lambda, torch.arange(length-1,-1,-1)).view(-1,1)
            MC_lmdas[epi_id][0:length] += last_step_lmda
        # TD LAMBDA RETURN
        Gt = (1 - self.TD_lambda) * torch.sum(lmdas * n_step_part, dim=-1, keepdim=True) +  MC_lmdas * Gt
        return Gt

    def _get_discounted_return(self, reward, bootstrap, terminate, epi_len):
        Gt = copy.deepcopy(reward)
        for epi_idx, epi_r in enumerate(Gt):
            end_step_idx = epi_len[epi_idx]-1
            if not terminate[epi_idx][end_step_idx]:
                # implement last step bootstrap
                # TODO
                epi_r[end_step_idx] += self.gamma * bootstrap[epi_idx][end_step_idx]
            for idx in range(end_step_idx-1, -1, -1):
                epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx+1]
        return Gt