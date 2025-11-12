import re
import torch
import torch.nn.functional as F
import random
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from models.MTLModel import _get_encoding
from utils import *


class Trainer:
    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params):
        # save arguments
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # Loss weights configuration
        self.lambda_recon = trainer_params.get('lambda_recon', 0.1)
        self.lambda_contrast = trainer_params.get('lambda_contrast', 0.05)
        self.lambda_guidance = trainer_params.get('lambda_guidance', 0.3)  # NEW: Diffusion guidance weight

        self.device = args.device
        self.log_path = args.log_path
        self.result_log = {"val_score": [], "val_gap": []}

        # Main Components
        self.envs = get_env(self.args.problem)
        self.model = get_model(self.args.model_type)(**self.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        num_param(self.model)

        # Restore
        self.start_epoch = 1
        if args.checkpoint is not None:
            checkpoint_fullname = args.checkpoint
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.start_epoch = 1 + checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = checkpoint['epoch'] - 1
            print(">> Checkpoint (Epoch: {}) Loaded!".format(checkpoint['epoch']))

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            print('=================================================================')

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.scheduler.step()

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            print("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['model_save_interval']
            
            if all_done or (epoch % model_save_interval == 0):
                print("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'problem': self.args.problem,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log
                }
                torch.save(checkpoint_dict, '{}/epoch-{}.pt'.format(self.log_path, epoch))

    def _train_one_epoch(self, epoch):
        episode = 0
        score_AM, loss_AM = AverageMeter(), AverageMeter()
        rl_loss_AM, guidance_loss_AM = AverageMeter(), AverageMeter()
        recon_loss_AM, contrast_loss_AM = AverageMeter(), AverageMeter()
        alpha_AM = AverageMeter()
        
        train_num_episode = self.trainer_params['train_episodes']

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            # Chọn ngẫu nhiên một môi trường để huấn luyện (MTL setup)
            env = random.sample(self.envs, 1)[0](**self.env_params)
            data = env.get_random_problems(batch_size, self.env_params["problem_size"])
            
            avg_score, loss_dict = self._train_one_batch(data, env)
            
            score_AM.update(avg_score, batch_size)
            loss_AM.update(loss_dict['total_loss'], batch_size)
            rl_loss_AM.update(loss_dict['rl_loss'], batch_size)
            guidance_loss_AM.update(loss_dict['guidance_loss'], batch_size)
            recon_loss_AM.update(loss_dict['recon_loss'], batch_size)
            contrast_loss_AM.update(loss_dict['contrast_loss'], batch_size)
            alpha_AM.update(loss_dict['alpha'], batch_size)
            
            episode += batch_size

        # Log Once, for each epoch
        print('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Total Loss: {:.4f}'.format(
            epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg))
        print('           RL: {:.4f}, Guidance: {:.4f}, Recon: {:.4f}, Contrast: {:.4f}, Alpha: {:.4f}'.format(
            rl_loss_AM.avg, guidance_loss_AM.avg, recon_loss_AM.avg, contrast_loss_AM.avg, alpha_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, data, env):
        self.model.train()
        self.model.set_eval_type(self.model_params["eval_type"])
        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        
        # Prep: Load problem and pre_forward
        env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        
        # Storage for episode trajectory
        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0), device=self.device)
        
        # MEMORY-EFFICIENT: Only store minimal info for guidance loss
        if self.model.use_diffusion_guidance:
            visited_mask_list = []  # Only store visited mask
            encoded_last_list = []  # Precompute encoded nodes
            attr_list = []  # Store attributes
            guidance_scores_list = []
            action_list = []  # Store actions
        
        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:
            current_selected_count = state.selected_count
            selected, prob, guidance_scores = self.model(state)
            # shape: (batch, pomo)
            
            # Store action
            if self.model.use_diffusion_guidance:
                if current_selected_count >= 2:
                   encoded_last_node = _get_encoding(self.model.encoded_nodes, state.current_node)
                   attr = torch.cat(
                         (state.load[:, :, None], state.current_time[:, :, None],
                           state.length[:, :, None], state.open[:, :, None]), 
                         dim=2
                    )
                   visited_mask = (state.ninf_mask == float('-inf'))
                   visited_mask_list.append(visited_mask.clone())
                   encoded_last_list.append(encoded_last_node.clone())
                   attr_list.append(attr.clone())
                   action_list.append(selected.clone())
                   guidance_scores_list.append(guidance_scores.clone())
            
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # ===================================================================
        # LOSS COMPUTATION
        # ===================================================================
        
        # 1. REINFORCE Loss (RL Loss)
        advantage = reward - reward.float().mean(dim=1, keepdims=True)  # (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        rl_loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        rl_loss_mean = rl_loss.mean()

        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        total_loss = rl_loss_mean
        
        # 2. Diffusion Guidance Loss (NEW)
        guidance_loss_mean = torch.tensor(0.0, device=self.device)
        if self.lambda_guidance > 0.0 and self.model.use_diffusion_guidance and len(action_list) > 0:
            try:
                # Compute baseline (average reward across pomo)
                baseline = reward.float().mean(dim=1, keepdims=True)  # (batch, 1)
                
                # Compute guidance loss for sampled steps (not all steps to save time)
                num_steps = len(action_list)
                sample_interval = max(1, num_steps // 20)  # Sample ~20 steps
                sampled_indices = range(0, num_steps, sample_interval)
                
                guidance_loss_list_final = []
                for step_idx in sampled_indices:
                    step_action = action_list[step_idx]
                    step_guidance_scores = guidance_scores_list[step_idx]
                    
                    # TÍNH LOSS TRỰC TIẾP TẠI ĐÂY
                    advantage = (reward - baseline).detach()
                    batch_size = step_action.size(0)
                    pomo_size = step_action.size(1)
                    batch_idx = torch.arange(batch_size, device=self.device)[:, None].expand(-1, pomo_size)
                    pomo_idx = torch.arange(pomo_size, device=self.device)[None, :].expand(batch_size, -1)
                    
                    # Compute guidance loss
                    selected_guidance = step_guidance_scores[batch_idx, pomo_idx, step_action]
                    step_guidance_loss = -(advantage * selected_guidance).mean()

                    guidance_loss_list_final.append(step_guidance_loss)
                
                # Average across sampled steps
                guidance_loss_mean = sum(guidance_loss_list_final) / len(guidance_loss_list_final)
                total_loss = total_loss + self.lambda_guidance * guidance_loss_mean
                
            except Exception as e:
                print(f"\nWarning: Guidance loss skipped due to error: {e}", end="")

        # 3. Slot Reconstruction Loss
        recon_loss_mean = torch.tensor(0.0, device=self.device)
        if self.lambda_recon > 0.0:
            try:
                recon_loss_mean = self.model.compute_slot_reconstruction_loss(reset_state)
                total_loss = total_loss + self.lambda_recon * recon_loss_mean
            except Exception as e:
                print(f"\nWarning: Reconstruction loss skipped due to error: {e}", end="")

        # 4. Slot Contrastive Loss
        contrast_loss_mean = torch.tensor(0.0, device=self.device)
        if self.lambda_contrast > 0.0:
            try:
                contrast_loss_mean = self.model.compute_slot_contrastive_loss()
                total_loss = total_loss + self.lambda_contrast * contrast_loss_mean
            except Exception as e:
                print(f"\nWarning: Contrastive loss skipped due to error: {e}", end="")

        # 5. Aux Loss (MoE Load Balancing, if exists)
        if hasattr(self.model, "aux_loss"):
            total_loss = total_loss + self.model.aux_loss

        # Backward and Step
        self.model.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (optional but recommended for diffusion)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        # Get current alpha value for logging
        current_alpha = 0.0
        if self.model.use_diffusion_guidance:
            current_alpha = torch.sigmoid(self.model.guidance_alpha).item()

        # Return loss dict
        loss_dict = {
            'total_loss': total_loss.item(),
            'rl_loss': rl_loss_mean.item(),
            'guidance_loss': guidance_loss_mean.item() if isinstance(guidance_loss_mean, torch.Tensor) else guidance_loss_mean,
            'recon_loss': recon_loss_mean.item() if isinstance(recon_loss_mean, torch.Tensor) else recon_loss_mean,
            'contrast_loss': contrast_loss_mean.item() if isinstance(contrast_loss_mean, torch.Tensor) else contrast_loss_mean,
            'alpha': current_alpha
        }

        return score_mean.item(), loss_dict

    def _clone_state(self, state):
        """
        Clone Step_State for guidance loss computation.
        Based on Step_State dataclass from CVRPEnv.
        
        NOTE: This function is now DEPRECATED in favor of memory-efficient approach.
        We now store minimal state info (visited_mask, encoded_last, attr) instead
        of cloning entire Step_State to save memory.
        """
        from dataclasses import fields
        
        # Create new Step_State instance
        cloned_state = type(state)()
        
        # Clone all fields
        for field in fields(state):
            val = getattr(state, field.name)
            if isinstance(val, torch.Tensor):
                setattr(cloned_state, field.name, val.clone())
            else:
                # For non-tensor fields (int, str, etc.), just copy reference
                setattr(cloned_state, field.name, val)
        
        return cloned_state

    def _val_one_batch(self, data, env, aug_factor=1, eval_type="argmax"):
        self.model.eval()
        self.model.set_eval_type(eval_type)
        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        with torch.no_grad():
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        max_pomo_reward, _ = aug_reward.max(dim=2)
        no_aug_score = -max_pomo_reward[0, :].float()
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
        aug_score = -max_aug_pomo_reward.float()

        return no_aug_score, aug_score

    def _val_and_stat(self, dir, val_path, env, batch_size=500, val_episodes=1000, compute_gap=False):
        no_aug_score_list, aug_score_list, no_aug_gap_list, aug_gap_list = [], [], [], []
        episode, no_aug_score, aug_score = 0, torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)

        while episode < val_episodes:
            remaining = val_episodes - episode
            bs = min(batch_size, remaining)
            data = env.load_dataset(os.path.join(dir, val_path), offset=episode, num_samples=bs)
            no_aug, aug = self._val_one_batch(data, env, aug_factor=8, eval_type="argmax")
            no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
            aug_score = torch.cat((aug_score, aug), dim=0)
            episode += bs

        no_aug_score_list.append(round(no_aug_score.mean().item(), 4))
        aug_score_list.append(round(aug_score.mean().item(), 4))

        if compute_gap:
            opt_sol = load_dataset(get_opt_sol_path(dir, env.problem, data[1].size(1)), disable_print=True)[: val_episodes]
            opt_sol = [i[0] for i in opt_sol]
            gap = [(no_aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
            no_aug_gap_list.append(round(sum(gap) / len(gap), 4))
            gap = [(aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
            aug_gap_list.append(round(sum(gap) / len(gap), 4))
            print(">> Val Score on {}: NO_AUG_Score: {}, NO_AUG_Gap: {}% --> AUG_Score: {}, AUG_Gap: {}%".format(
                val_path, no_aug_score_list, no_aug_gap_list, aug_score_list, aug_gap_list))
            return aug_score_list[0], aug_gap_list[0]
        else:
            print(">> Val Score on {}: NO_AUG_Score: {}, --> AUG_Score: {}".format(
                val_path, no_aug_score_list, aug_score_list))
            return aug_score_list[0], 0