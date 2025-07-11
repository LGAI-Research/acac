import time
import numpy as np
import os
import wandb

from acac.acac_marl.cores.acac.memory import Memory_epi
from acac.acac_marl.cores.acac.envs_runner import EnvsRunner
from acac.acac_marl.cores.acac.utils import Linear_Decay, save_train_data, save_test_data, save_policies
from acac.acac_marl.cores.acac.ckpt_utils import save_checkpoint_cent, load_checkpoint_cent
from acac.acac_marl.cores.acac.controller import MAC
from acac.acac_marl.cores.acac import Learner_ACAC
from acac.acac_marl.misc.wandb_utils import log_stat_wandb
from acac.acac_marl.misc.logger import logger
from acac.acac_marl import PROJECT_DIR

class AgentCentricActorCritic(object):

    def __init__(self,
            env,
            env_terminate_step, 
            n_env, 
            n_agent, 
            seed, 
            save_dir, 
            resume, 
            device, 
            total_epi, 
            gamma, 
            a_lr, 
            c_lr, 
            c_train_iteration, 
            eps_start, 
            eps_end, 
            eps_stable_at, 
            c_hys_start, 
            c_hys_end, 
            adv_hys_start, 
            adv_hys_end, 
            hys_stable_at, 
            critic_hys, 
            adv_hys, 
            etrpy_w_start, 
            etrpy_w_end, 
            etrpy_w_stable_at, 
            train_freq, 
            c_target_update_freq, 
            c_target_soft_update, 
            n_train_repeat,
            n_minibatch,
            tau, 
            n_step_TD, 
            TD_lambda, 
            GAE_lambda,
            a_mlp_layer_size, 
            a_rnn_layer_size, 
            c_mlp_layer_size, 
            c_rnn_layer_size, 
            grad_clip_value, 
            grad_clip_norm, 
            obs_last_action, 
            eval_policy, 
            eval_freq, 
            eval_num_epi, 
            sample_epi, 
            trace_len, 
            time_emb,
            time_emb_actor,
            clip_ratio,
            parallel,
            share_encoder,
            use_attention,
            cc_n_head,
            enc_n_head,
            n_layer,
            value_head,
            time_emb_alg,
            time_emb_dim,
            state,
            use_actor_ln,
            duplicate,
            max_checkpoint=10,
            vf_coef=0.5,
            use_popart=False,
            wandb=False,
            *args, 
            **kwargs):

        self.total_epi = total_epi
        self.train_freq = train_freq
        self.eval_policy = eval_policy
        self.eval_freq = eval_freq
        self.eval_num_epi = eval_num_epi
        self.critic_hys = critic_hys
        self.adv_hys = adv_hys 
        self.sample_epi = sample_epi
        self.c_target_update_freq = c_target_update_freq
        self.c_target_soft_update = c_target_soft_update
        self.save_dir = save_dir
        self.seed = seed
        self.resume = resume
        self.state = state
        self.duplicate = duplicate
        
        self.parallel = parallel
        self.share_encoder = share_encoder
        self.use_attention = use_attention
        self.value_head = value_head
        self.cc_n_head = cc_n_head
        self.enc_n_head = enc_n_head
        self.n_layer = n_layer

        self.total_timesteps = 0
        self.max_timesteps = env_terminate_step * self.total_epi

        self.total_time = 0
        self.epoch_time = 0
        self.sampling_time = 0
        self.training_time = 0
        
        self.time_emb = time_emb
        self.time_emb_actor = time_emb_actor
        self.time_emb_alg = time_emb_alg
        self.time_emb_dim = time_emb_dim
        self.use_actor_ln = use_actor_ln

        self.max_checkpoint = max_checkpoint
        self.wandb = wandb
        

        # collect params
        actor_params = {'a_mlp_layer_size': a_mlp_layer_size,
                        'a_rnn_layer_size': a_rnn_layer_size}

        critic_params = {'c_mlp_layer_size': c_mlp_layer_size,
                         'c_rnn_layer_size': c_rnn_layer_size,
                         'use_popart': use_popart}
        hyper_params = {'a_lr': a_lr,
                        'c_lr': c_lr,
                        'c_train_iteration': c_train_iteration,
                        'c_target_update_freq': c_target_update_freq,
                        'n_train_repeat': n_train_repeat,
                        'n_minibatch': n_minibatch,
                        'tau': tau,
                        'grad_clip_value': grad_clip_value,
                        'grad_clip_norm': grad_clip_norm,
                        'n_step_TD': n_step_TD,
                        'TD_lambda': TD_lambda,
                        'GAE_lambda': GAE_lambda,
                        'device': device,
                        'clip_ratio': clip_ratio,
                        'vf_coef': vf_coef}

        self.env = env
        # create buffer
        self.memory = Memory_epi(env.obs_size, env.n_action, obs_last_action, size=train_freq)
        self.buffer = [] # To save whole memory
            
        # cretate controller
        self.controller = MAC(
            self.env, obs_last_action, **actor_params, **critic_params, device=device,
            time_emb=self.time_emb, time_emb_actor=self.time_emb_actor, init_critic=False,
            share_encoder=self.share_encoder, use_attention=self.use_attention, 
            cc_n_head=self.cc_n_head, enc_n_head=self.enc_n_head, n_layer=self.n_layer,
            value_head=self.value_head, time_emb_alg = self.time_emb_alg,
            time_emb_dim=self.time_emb_dim, max_timestep=env_terminate_step, 
            use_actor_ln=self.use_actor_ln, duplicate=self.duplicate,
        )
        # create parallel envs runner
        self.envs_runner = EnvsRunner(self.env, n_env, self.controller, self.memory, env_terminate_step, gamma, seed, obs_last_action, trace_len, parallel)
        
        # create learner
        self.learner = Learner_ACAC(
            self.env, self.controller, self.memory, gamma, obs_last_action, 
            **hyper_params, **critic_params
        )
        self.save_fn = save_checkpoint_cent
        self.load_fn = load_checkpoint_cent
            
        
        # create epsilon calculator for implementing e-greedy exploration policy
        self.eps_call = Linear_Decay(eps_stable_at, eps_start, eps_end)
        # create hysteretic calculator for implementing hystgeritic value function updating
        self.c_hys_call = Linear_Decay(hys_stable_at, c_hys_start, c_hys_end)
        # create hysteretic calculator for implementing hystgeritic advantage esitimation
        self.adv_hys_call = Linear_Decay(hys_stable_at, adv_hys_start, adv_hys_end)
        # create entropy loss weight calculator
        self.etrpy_w_call = Linear_Decay(etrpy_w_stable_at, etrpy_w_start, etrpy_w_end)
        # record evaluation return
        self.eval_returns = []

    def learn(self):
        epi_count = 0
        if self.resume:
            epi_count, self.eval_returns = self.load_fn(self.seed, self.save_dir, self.controller, self.learner, self.envs_runner)

        epoch_start_time = time.time()
        save_checkpoint_ind = 0
        timestep_gap_between_checkpoints = int(self.max_timesteps // self.max_checkpoint)
        while self.total_timesteps < self.max_timesteps:
            # update eps
            eps = self.eps_call.get_value(epi_count)
            # update hys
            c_hys_value = self.c_hys_call.get_value(epi_count)
            adv_hys_value = self.adv_hys_call.get_value(epi_count)
            # update etrpy weight
            etrpy_w = self.etrpy_w_call.get_value(epi_count)
            # let envs run a certain number of episodes accourding to train_freq
            sampling_start = time.time()
            self.envs_runner.run(eps=eps, n_epis=self.train_freq)
            self.sampling_time += time.time() - sampling_start

            # perform hysteretic-ac update
            train_start = time.time()
            self.learner.train(eps, c_hys_value, adv_hys_value, etrpy_w, self.critic_hys, self.adv_hys)
            self.training_time += time.time() - train_start
            if not self.sample_epi:
                self.memory.buf.clear()

            epi_count += self.train_freq

            # update target net
            if self.c_target_soft_update:
                self.learner.update_critic_target_net(soft=True)
                self.learner.update_actor_target_net(soft=True)
            elif epi_count % self.c_target_update_freq == 0:
                self.learner.update_critic_target_net()
                self.learner.update_actor_target_net()

            if self.eval_policy and epi_count % (self.eval_freq - (self.eval_freq % self.train_freq)) == 0:
                self.envs_runner.run(n_epis=self.eval_num_epi, test_mode=True)
                assert len(self.envs_runner.eval_returns) >= self.eval_num_epi, "Not evaluate enough episodes ..."
                self.eval_returns.append(np.mean(self.envs_runner.eval_returns[-self.eval_num_epi:]))
                self.envs_runner.eval_returns = []
                
                self.epoch_time = time.time() - epoch_start_time
                self.total_time += self.epoch_time

                logger.log(f'Episode {epi_count} | Eval Finished')
                self.log_diagnostics(epi_count, eps)

                # save the best policy
                if self.eval_returns[-1] == np.max(self.eval_returns):
                    save_policies(self.seed, self.controller.agents, self.save_dir)

                epoch_start_time = time.time()
                self.sampling_time = 0
                self.training_time = 0
            
            ################################ saving in the middle ###################################
            if self.total_timesteps > (save_checkpoint_ind + 1) * timestep_gap_between_checkpoints:
                self.save_fn(self.seed, epi_count, self.eval_returns, self.controller, self.learner, self.envs_runner, self.save_dir, max_save=self.max_checkpoint+1)
                save_checkpoint_ind += 1

        ################################ saving in the end ###################################
        self.save_fn(self.seed, epi_count, self.eval_returns, self.controller, self.learner, self.envs_runner, self.save_dir, max_save=self.max_checkpoint+1)
        self.envs_runner.close()

        print(f"{[self.seed]} Finish entire training ... ", flush=True)

    def log_diagnostics(self, epi_count, eps):
        wandb_log_stats = {}

        runner_diag = self.envs_runner.get_diagnostics()
        for k, v in runner_diag.items():
            if 'Agent' in k:
                if 'NMacroActions' in k:
                    logger.record_tabular(k, np.mean(v))
                    wandb_log_stats[k] = np.mean(v)
            else:
                logger.record_tabular_misc_stat(k, v)
        
        wandb_log_stats['Train/Return'] = np.mean(runner_diag['Train/Return'])
        wandb_log_stats['Train/EpiLen'] = np.mean(runner_diag['Train/EpiLen'])
        wandb_log_stats['Eval/Return'] = np.mean(runner_diag['Eval/Return'])
        wandb_log_stats['Eval/EpiLen'] = np.mean(runner_diag['Eval/EpiLen'])
        for prefix in ["Train/", "Eval/"]:
            for k in ["NumDelivery", "PickRate"]:
                key = prefix + k
                if key in runner_diag:
                    wandb_log_stats[key] = np.mean(runner_diag[key])
        
        self.total_timesteps += np.sum(runner_diag['Train/EpiLen'])

        learner_diag = self.learner.get_diagnostics()
        for k, v in learner_diag.items():
            logger.record_tabular(k, np.mean(v))
            wandb_log_stats[k] = np.mean(v)

        logger.record_tabular('Eps', eps)
        logger.record_tabular('Episodes', epi_count)
        logger.record_tabular('Timesteps', self.total_timesteps)
        logger.record_tabular('SamplingTime', self.sampling_time)
        logger.record_tabular('TrainingTime', self.training_time)
        logger.record_tabular('EpochTime', self.epoch_time)
        logger.record_tabular('TotalTime', self.total_time)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        wandb_log_stats['Timesteps'] = self.total_timesteps
        wandb_log_stats['Episodes'] = epi_count
        if self.wandb:
            log_stat_wandb(wandb_log_stats, step=self.total_timesteps)


