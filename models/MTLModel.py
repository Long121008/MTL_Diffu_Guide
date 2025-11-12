import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['MTLModel']


# =========================================================================
# DIFFUSION GUIDANCE MODULE (NEW)
# =========================================================================

class DiffusionGuidance(nn.Module):
    """
    Diffusion-based guidance module for autoregressive decoder.
    Learns to predict good next-node scores via denoising process.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.num_timesteps = model_params.get('diffusion_timesteps', 50)
        self.head_num = model_params['head_num']
        
        # Time embedding (sinusoidal like DDPM)
        time_dim = 128
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # State encoder: encode current partial tour
        self.state_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.head_num,
                dim_feedforward=self.embedding_dim * 4,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Attribute encoder: encode current load, time, etc.
        self.attr_encoder = nn.Sequential(
            nn.Linear(4, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # Denoising network (U-Net style transformer)
        self.denoiser = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.embedding_dim,
                nhead=self.head_num,
                dim_feedforward=self.embedding_dim * 4,
                batch_first=True
            ) for _ in range(3)
        ])
        
        # Output head: predict guidance scores
        self.score_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, 1)
        )
        
        # Noise schedule (cosine schedule)
        self.register_buffer('betas', self._cosine_beta_schedule(self.num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - self.alphas_cumprod))
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in Improved DDPM"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _get_timestep_embedding(self, timesteps, dim=128):
        """Sinusoidal timestep embeddings"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb
    
    def forward(self, current_state, encoded_nodes, slots, ninf_mask=None, 
                num_inference_steps=None):
        """
        Args:
            current_state: dict with keys:
                - 'visited_mask': [B, P, N] boolean tensor
                - 'encoded_last': [B, P, D] last visited node encoding
                - 'attr': [B, P, 4] current load, time, length, open
            encoded_nodes: [B, N, D] all node encodings
            slots: [B, K, D] slot features for global context
            ninf_mask: [B, P, N] mask for infeasible nodes
            num_inference_steps: int, number of denoising steps (default: self.num_timesteps)
        
        Returns:
            guidance_scores: [B, P, N] guidance scores for each node
        """
        B, N, D = encoded_nodes.shape
        P = current_state['encoded_last'].size(1)
        device = encoded_nodes.device
        
        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps
        
        # === 1. Encode current state (100% TENSORIZED - NO LOOPS!) ===
        visited_mask = current_state['visited_mask']  # [B, P, N]
        
        # Expand encoded_nodes for all pomo instances: [B, N, D] -> [B*P, N, D]
        encoded_nodes_bp = encoded_nodes.unsqueeze(1).expand(B, P, N, D).reshape(B * P, N, D)
        
        # Flatten visited mask: [B, P, N] -> [B*P, N]
        visited_mask_bp = visited_mask.reshape(B * P, N)
        
        # Get visited lengths for each B*P instance
        visited_lengths = visited_mask_bp.sum(dim=1)  # [B*P]
        max_visited = visited_lengths.max().item()
        if max_visited == 0:
            max_visited = 1  # Avoid empty sequence
        
        # === 100% TENSORIZED GATHERING (OPTIMIZED) ===
        # Strategy: Use advanced indexing with cumsum to create gathering indices
        
        # Step 1: Get positions of True values in visited_mask
        # For each row in [B*P], get node indices where visited_mask is True
        # Result will be a 1D tensor of all visited node indices (flattened)
        bp_indices, node_indices = visited_mask_bp.nonzero(as_tuple=True)
        # bp_indices: which batch*pomo instance [0 to B*P-1]
        # node_indices: which node was visited [0 to N-1]
        
        # Step 2: Compute position within each sequence using cumsum trick
        # For each instance in bp_indices, we need to know "this is the k-th visited node"
        # Create a counter that resets at each new bp instance
        bp_changes = torch.cat([
            torch.tensor([0], device=device),
            (bp_indices[1:] != bp_indices[:-1]).long()
        ])
        position_in_seq = torch.cumsum(1 - bp_changes, dim=0) - 1  # 0-indexed position
        
        # Step 3: Create output tensor [B*P, max_visited, D]
        visited_embeds = torch.zeros(B * P, max_visited, D, device=device)
        
        # Step 4: Gather embeddings using advanced indexing
        # Get all visited embeddings at once
        visited_node_embeds = encoded_nodes_bp[bp_indices, node_indices]  # [num_visited_total, D]
        
        # Step 5: Scatter into the output tensor
        # Filter out positions beyond max_visited (shouldn't happen, but safe)
        valid_positions = position_in_seq < max_visited
        bp_indices_valid = bp_indices[valid_positions]
        position_in_seq_valid = position_in_seq[valid_positions]
        visited_node_embeds_valid = visited_node_embeds[valid_positions]
        
        # Scatter: visited_embeds[bp_idx, pos_in_seq] = embedding
        visited_embeds[bp_indices_valid, position_in_seq_valid] = visited_node_embeds_valid
        
        # visited_embeds now has shape [B*P, max_visited, D] with proper padding
        # No masking needed - zero padding is already correct
        
        # Encode tour state with TransformerEncoder
        tour_state = self.state_encoder(visited_embeds)  # [B*P, max_visited, D]
        tour_state = tour_state.mean(dim=1)  # [B*P, D] - average pooling
        tour_state = tour_state.reshape(B, P, D)
        
        # Encode attributes
        attr_embed = self.attr_encoder(current_state['attr'])  # [B, P, D]
        
        # Combine state representations
        state_context = tour_state + attr_embed  # [B, P, D]
        
        # === 2. Prepare condition (context for denoising) ===
        # Expand encoded_nodes for each pomo
        encoded_nodes_expanded = encoded_nodes.unsqueeze(1).expand(B, P, N, D)  # [B, P, N, D]
        
        # Expand slots for each pomo
        K = slots.size(1)
        slots_expanded = slots.unsqueeze(1).expand(B, P, K, D)  # [B, P, K, D]
        
        # Concatenate: state + slots + nodes as condition
        # Reshape for transformer: [B*P, 1+K+N, D]
        condition = torch.cat([
            state_context.reshape(B*P, 1, D),
            slots_expanded.reshape(B*P, K, D),
            encoded_nodes_expanded.reshape(B*P, N, D)
        ], dim=1)
        
        # === 3. Diffusion denoising process ===
        # Initialize with pure noise
        x_t = torch.randn(B*P, N, D, device=device)
        
        # Sampling schedule (can use DDIM for fewer steps)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, 
                                  dtype=torch.long, device=device)
        
        for t_idx in timesteps:
            t = t_idx.unsqueeze(0).expand(B*P)
            
            # Time embedding
            t_embed = self._get_timestep_embedding(t)  # [B*P, 128]
            t_embed = self.time_embed(t_embed)  # [B*P, D]
            
            # Add time info to noisy features
            x_t_with_time = x_t + t_embed.unsqueeze(1)  # [B*P, N, D]
            
            # Denoise with transformer decoder
            denoised = x_t_with_time
            for layer in self.denoiser:
                denoised = layer(denoised, condition)  # [B*P, N, D]
            
            # Predict noise (epsilon prediction)
            noise_pred = denoised - x_t
            
            # DDPM update
            if t_idx > 0:
                alpha_t = self.alphas_cumprod[t_idx]
                alpha_t_prev = self.alphas_cumprod[t_idx - 1]
                beta_t = self.betas[t_idx]
                
                # Predict x_0
                x_0_pred = (x_t - self.sqrt_one_minus_alphas_cumprod[t_idx] * noise_pred) / \
                          self.sqrt_alphas_cumprod[t_idx]
                
                # Add noise for next step
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_t_prev) * x_0_pred + \
                      torch.sqrt(1 - alpha_t_prev) * noise
            else:
                # Final step
                x_t = (x_t - self.sqrt_one_minus_alphas_cumprod[t_idx] * noise_pred) / \
                      self.sqrt_alphas_cumprod[t_idx]
        
        # === 4. Predict final guidance scores ===
        x_0 = x_t  # [B*P, N, D]
        guidance_scores = self.score_head(x_0).squeeze(-1)  # [B*P, N]
        guidance_scores = guidance_scores.reshape(B, P, N)
        
        # Apply mask if provided
        if ninf_mask is not None:
            guidance_scores = guidance_scores + ninf_mask
        
        return guidance_scores


# =========================================================================
# SLOT ATTENTION MODULES
# =========================================================================

class IterativeAttention(nn.Module):
    def __init__(self, embedding_dim, slot_dim, head_num, qkv_dim):
        super().__init__()
        self.scale = qkv_dim ** -0.5
        
        self.Wq = nn.Linear(slot_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, slot_dim)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm = nn.LayerNorm(slot_dim)
        
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        
    def forward(self, slot_features, node_features, return_attention=False):
        B, N, D_emb = node_features.shape
        K_total, D_slot = slot_features.shape
        K = K_total // B

        q = reshape_by_heads(self.Wq(slot_features).reshape(B, K, -1), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(node_features), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(node_features), head_num=self.head_num)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        weights = F.softmax(scores, dim=-1)

        attention_map = weights.mean(dim=1) if return_attention else None

        slot_updates = torch.matmul(weights, v)
        slot_updates = slot_updates.transpose(1, 2).reshape(B, K, -1)
        slot_updates = self.multi_head_combine(slot_updates).reshape(B * K, D_slot)

        slots_normalized = self.norm(slot_features)
        slots_updated = self.gru(slot_updates, slots_normalized)

        if return_attention:
            return slots_updated, attention_map
        return slots_updated


class SlotAttentionModule(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        slot_num = model_params.get('slot_num', 16)
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']
        
        self.slot_num = slot_num
        self.iter_num = model_params.get('slot_iter_num', 3)
        self.embedding_dim = embedding_dim
        
        self.slots_mu = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        self.iterative_attention = IterativeAttention(embedding_dim, embedding_dim, head_num, qkv_dim)
        
        self.last_slots = None
        self.last_attention = None

    def forward(self, node_features):
        B, N, D = node_features.shape
        
        mu = self.slots_mu.expand(B, self.slot_num, D)
        log_sigma = self.slots_log_sigma.expand(B, self.slot_num, D)
        sigma = torch.exp(log_sigma)
        
        eps = torch.randn_like(mu)
        initial_slots = mu + sigma * eps
        
        slots = initial_slots.reshape(B * self.slot_num, D)
        
        attention_maps = []
        for i in range(self.iter_num):
            if i == self.iter_num - 1:
                slots, attn = self.iterative_attention(slots, node_features, return_attention=True)
                attention_maps.append(attn)
            else:
                slots = self.iterative_attention(slots, node_features, return_attention=False)
        
        slots_out = slots.reshape(B, self.slot_num, D)
        
        self.last_slots = slots_out
        self.last_attention = attention_maps[-1] if attention_maps else None

        return slots_out


# =========================================================================
# MTL MODEL
# =========================================================================

class MTLModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']

        self.encoder = MTL_Encoder(**model_params)
        self.decoder = MTL_Decoder(**model_params)
        
        # Diffusion Guidance (NEW)
        self.use_diffusion_guidance = model_params.get('use_diffusion_guidance', False)
        if self.use_diffusion_guidance:
            self.diffusion_guidance = DiffusionGuidance(**model_params)
            # Learnable weight for guidance (starts small)
            self.guidance_alpha = nn.Parameter(torch.tensor(0.1))
        
        self.enable_reconstruction = model_params.get('enable_slot_reconstruction', False)
        if self.enable_reconstruction:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(model_params['embedding_dim'], 256),
                nn.ReLU(),
                nn.Linear(256, 5)
            )
        
        self.encoded_nodes = None
        self.slots = None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_tw_start = reset_state.node_tw_start
        node_tw_end = reset_state.node_tw_end
        
        node_xy_demand_tw = torch.cat(
            (node_xy, node_demand[:, :, None], 
             node_tw_start[:, :, None], node_tw_end[:, :, None]), 
            dim=2
        )

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_tw)
        self.slots = self.encoder.slot_attention_module.last_slots
        
        self.decoder.set_kv(self.encoded_nodes, slots=self.slots)

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, selected=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        guidance_scores_to_return = None

        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long).to(self.device)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:
            selected = state.START_NODE
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            attr = torch.cat(
                (state.load[:, :, None], state.current_time[:, :, None],
                 state.length[:, :, None], state.open[:, :, None]), 
                dim=2
            )
            
            # AR decoder probabilities
            probs = self.decoder(encoded_last_node, attr, ninf_mask=state.ninf_mask)
            guidance_scores_to_return = None

            # === DIFFUSION GUIDANCE (NEW) ===
            if self.use_diffusion_guidance:
                # Prepare current state for guidance
                visited_mask = (state.ninf_mask == float('-inf'))
                current_state = {
                    'visited_mask': visited_mask,  # Assuming this exists in state
                    'encoded_last': encoded_last_node,
                    'attr': attr
                }
                
                # Get guidance scores
                guidance_scores = self.diffusion_guidance(
                    current_state=current_state,
                    encoded_nodes=self.encoded_nodes,
                    slots=self.slots,
                    ninf_mask=state.ninf_mask,
                    num_inference_steps=10 if self.training else 20  # Fewer steps during training
                )
                guidance_scores_to_return = guidance_scores.clone()
                # Combine AR logits with guidance
                alpha = torch.sigmoid(self.guidance_alpha)  # Clamp to [0,1]
                ar_logits = torch.log(probs + 1e-10)
                combined_logits = ar_logits + alpha * guidance_scores
                
                # Apply mask and get final probs
                combined_logits = combined_logits + state.ninf_mask
                probs = F.softmax(combined_logits, dim=2)
            
            if selected is None:
                while True:
                    if self.training or self.eval_type == 'softmax':
                        try:
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                                .squeeze(dim=1).reshape(batch_size, pomo_size)
                        except Exception as exception:
                            print(f">> Catch Exception: {exception}, on instances of {state.PROBLEM}")
                            exit(0)
                    else:
                        selected = probs.argmax(dim=2)
                    
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        return selected, prob, guidance_scores_to_return
    
    def compute_guidance_loss(self, state, selected_action, reward, baseline):
        """
        Compute guidance loss for REINFORCE training.
        Guides diffusion to predict actions with higher rewards.
        
        Args:
            state: can be either:
                - Step_State object (full state)
                - dict with keys: 'visited_mask', 'encoded_last', 'attr' (minimal state)
            selected_action: [B, P] selected actions
            reward: [B, P] rewards for current trajectory
            baseline: [B, P] or [B, 1] baseline for variance reduction
        
        Returns:
            guidance_loss: scalar tensor
        """
        if not self.use_diffusion_guidance:
            return torch.tensor(0.0, device=self.device)
        
        # Handle both full state and minimal state dict
        if isinstance(state, dict):
            # Minimal state (memory efficient)
            visited_mask = state['visited_mask']
            encoded_last_node = state['encoded_last']
            attr = state['attr']
            
            current_state = {
                'visited_mask': visited_mask,
                'encoded_last': encoded_last_node,
                'attr': attr
            }
        else:
            # Full Step_State object
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            attr = torch.cat(
                (state.load[:, :, None], state.current_time[:, :, None],
                 state.length[:, :, None], state.open[:, :, None]), 
                dim=2
            )
            
            # Create visited mask from ninf_mask
            visited_mask = (state.ninf_mask == float('-inf'))
            
            current_state = {
                'visited_mask': visited_mask,
                'encoded_last': encoded_last_node,
                'attr': attr
            }
        
        # Get guidance scores
        guidance_scores = self.diffusion_guidance(
            current_state=current_state,
            encoded_nodes=self.encoded_nodes,
            slots=self.slots,
            ninf_mask=None,  # Already handled in visited_mask
            num_inference_steps=5  # Very few steps for training efficiency
        )
        
        # Supervised signal: guide towards actions with high advantage
        advantage = (reward - baseline).detach()  # [B, P] or broadcast from [B, 1]
        
        # Get guidance scores for selected actions
        batch_size = selected_action.size(0)
        pomo_size = selected_action.size(1)
        batch_idx = torch.arange(batch_size, device=self.device)[:, None].expand(-1, pomo_size)
        pomo_idx = torch.arange(pomo_size, device=self.device)[None, :].expand(batch_size, -1)
        
        selected_guidance = guidance_scores[batch_idx, pomo_idx, selected_action]  # [B, P]
        
        # Loss: maximize guidance for good actions (positive advantage)
        # Use advantage as weight: higher advantage → stronger signal
        guidance_loss = -(advantage * selected_guidance).mean()
        
        return guidance_loss
    
    def compute_slot_reconstruction_loss(self, reset_state):
        if not self.enable_reconstruction or self.slots is None:
            return torch.tensor(0.0, device=self.device)
        
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_tw_start = reset_state.node_tw_start
        node_tw_end = reset_state.node_tw_end
        
        original_features = torch.cat(
            (node_xy, node_demand[:, :, None],
             node_tw_start[:, :, None], node_tw_end[:, :, None]),
            dim=2
        )
        
        attention = self.encoder.slot_attention_module.last_attention
        attention_customers = attention[:, :, 1:]
        
        slot_predictions = self.reconstruction_head(self.slots)
        
        reconstructed_features = torch.einsum(
            'bkn,bkf->bnf',
            attention_customers,
            slot_predictions
        )
        
        recon_loss = F.mse_loss(reconstructed_features, original_features)
        
        return recon_loss
    
    def compute_slot_contrastive_loss(self):
        if self.slots is None:
            return torch.tensor(0.0, device=self.device)
        
        slots_norm = F.normalize(self.slots, dim=-1)
        similarity = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
        
        batch, K, _ = similarity.shape
        mask = torch.eye(K, device=self.device).unsqueeze(0).expand(batch, -1, -1)
        off_diag_sim = similarity * (1 - mask)
        
        contrastive_loss = (off_diag_sim ** 2).sum() / (batch * K * (K - 1))
        
        return contrastive_loss


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_nodes


# =========================================================================
# ENCODER
# =========================================================================

class MTL_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)
        
        self.slot_attention_module = SlotAttentionModule(**model_params)
        self.slot_to_node_cross_atten = EncoderLayer(**model_params)
        
        self.layers = nn.ModuleList([
            EncoderLayer(**model_params) for _ in range(encoder_layer_num)
        ])

    def forward(self, depot_xy, node_xy_demand_tw):
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand_tw)

        H_nodes = torch.cat((embedded_depot, embedded_node), dim=1)
        H_slots = self.slot_attention_module(H_nodes)
        H_reconstructed = self.slot_to_node_cross_atten.forward_cross(H_nodes, H_slots)
        
        out = H_reconstructed
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)
            multi_head_out = self.multi_head_combine(out_concat)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)
        else:
            out1 = self.addAndNormalization1(None, input1)
            out_concat = multi_head_attention(q, k, v)
            multi_head_out = self.multi_head_combine(out_concat)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2 = self.feedForward(out2)
            out3 = input2 + out2

        return out3
    
    def forward_cross(self, query_input, kv_input):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(query_input), head_num=head_num)
        k = reshape_by_heads(self.Wk(kv_input), head_num=head_num)
        v = reshape_by_heads(self.Wv(kv_input), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        if self.model_params['norm_loc'] == "norm_last":
            out1 = self.addAndNormalization1(query_input, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)
        else:
            out1 = self.addAndNormalization1(None, query_input)
            input2 = out1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2 = self.feedForward(out2)
            out3 = input2 + out2

        return out3


# =========================================================================
# DECODER
# =========================================================================

class MTL_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        
        self.Wk_nodes = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_nodes = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        self.Wk_slots = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_slots = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.slot_gate = nn.Linear(embedding_dim + 4, 1)

        self.k_nodes = None
        self.v_nodes = None
        self.k_slots = None
        self.v_slots = None
        self.single_head_key_nodes = None
        self.slots = None

    def set_kv(self, encoded_nodes, slots=None):
        head_num = self.model_params['head_num']

        self.k_nodes = reshape_by_heads(self.Wk_nodes(encoded_nodes), head_num=head_num)
        self.v_nodes = reshape_by_heads(self.Wv_nodes(encoded_nodes), head_num=head_num)
        self.single_head_key_nodes = encoded_nodes.transpose(1, 2)
        
        if slots is not None:
            self.slots = slots
            self.k_slots = reshape_by_heads(self.Wk_slots(slots), head_num=head_num)
            self.v_slots = reshape_by_heads(self.Wv_slots(slots), head_num=head_num)

    def forward(self, encoded_last_node, attr, ninf_mask):
        head_num = self.model_params['head_num']

        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)

        out_concat_nodes = multi_head_attention(q_last, self.k_nodes, self.v_nodes,
                                               rank3_ninf_mask=ninf_mask)

        if self.slots is not None:
            out_concat_slots = multi_head_attention(q_last, self.k_slots, self.v_slots)
            
            gate_logit = self.slot_gate(input_cat)
            gate_weight = torch.sigmoid(gate_logit)
            
            out_concat = gate_weight * out_concat_slots + (1 - gate_weight) * out_concat_nodes
        else:
            out_concat = out_concat_nodes

        mh_atten_out = self.multi_head_combine(out_concat)

        score = torch.matmul(mh_atten_out, self.single_head_key_nodes)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)

        return probs


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if 'norm_loc' in model_params.keys() and model_params['norm_loc'] == "norm_last" else False
        
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            normalized = self.norm(transposed)
            back_trans = normalized.transpose(1, 2)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))