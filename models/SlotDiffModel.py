import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SlotDiffModel']


# =========================================================================
# SLOT ATTENTION MODULES (PLUG-AND-PLAY)
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
# LATENT DIFFUSION DENOISER (PLUG-AND-PLAY)
# =========================================================================

class LDMDenoiser(nn.Module):
    """Denoiser for Slot Diffusion"""
    def __init__(self, feature_dim, slot_dim, denoiser_dim, n_heads=4, n_layers=2, max_timesteps=1000):
        super().__init__()
        
        # Input/Output Projections
        self.input_projection = nn.Linear(feature_dim, denoiser_dim)
        
        # Time Embedding
        self.time_embedder = nn.Sequential(
            nn.Linear(denoiser_dim, denoiser_dim),
            nn.SiLU(),
            nn.Linear(denoiser_dim, denoiser_dim)
        )
        self.denoiser_dim = denoiser_dim
        
        # Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=denoiser_dim,
            kdim=slot_dim,
            vdim=slot_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        self.cross_norm = nn.LayerNorm(denoiser_dim)
        
        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=denoiser_dim,
                nhead=n_heads,
                dim_feedforward=denoiser_dim * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layers)
        ])
        
        # Output Layer
        self.out_layer = nn.Sequential(
            nn.Linear(denoiser_dim, denoiser_dim * 2),
            nn.SiLU(),
            nn.Linear(denoiser_dim * 2, feature_dim)
        )

    def _get_timestep_embedding(self, timesteps, dim):
        import math
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, z_t, t, slots):
        B, N, D = z_t.shape
        
        # Project z_t
        z_t_embed = self.input_projection(z_t)

        # Embed timestep
        t_emb_sin = self._get_timestep_embedding(t, self.denoiser_dim)
        t_emb = self.time_embedder(t_emb_sin)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)
        
        # Add time
        z_t_with_time = z_t_embed + t_emb
        
        # Cross-Attention
        attn_output, _ = self.cross_attention(
            query=z_t_with_time,
            key=slots,
            value=slots
        )
        
        # Residual + Norm
        x = self.cross_norm(z_t_with_time + attn_output)
        
        # Transformer Layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Predict noise
        predicted_noise = self.out_layer(x)
        
        return predicted_noise


# =========================================================================
# ENHANCED ENCODER LAYER (PROGRESSIVE SLOT INTEGRATION)
# =========================================================================

class EnhancedEncoderLayer(nn.Module):
    """Encoder layer with optional slot cross-attention"""
    def __init__(self, depth=0, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']
        
        # Self-Attention
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        # Cross-Attention with Slots
        self.Wq_cross = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_cross = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_cross = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine_cross = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.addAndNormalization_cross = Add_And_Normalization_Module(**model_params)
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1, slot_features=None):
        head_num = self.model_params['head_num']
        
        # Self-Attention
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        
        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)
            multi_head_out = self.multi_head_combine(out_concat)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            
            # Cross-Attention with Slots
            if slot_features is not None:
                q_cross = reshape_by_heads(self.Wq_cross(out1), head_num=head_num)
                k_cross = reshape_by_heads(self.Wk_cross(slot_features), head_num=head_num)
                v_cross = reshape_by_heads(self.Wv_cross(slot_features), head_num=head_num)
                out_concat_cross = multi_head_attention(q_cross, k_cross, v_cross)
                multi_head_out_cross = self.multi_head_combine_cross(out_concat_cross)
                out1 = self.addAndNormalization_cross(out1, multi_head_out_cross)
            
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)
        else:
            out1 = self.addAndNormalization1(None, input1)
            out_concat = multi_head_attention(q, k, v)
            multi_head_out = self.multi_head_combine(out_concat)
            input2 = input1 + multi_head_out
            
            # Cross-Attention with Slots
            if slot_features is not None:
                q_cross = reshape_by_heads(self.Wq_cross(input2), head_num=head_num)
                k_cross = reshape_by_heads(self.Wk_cross(slot_features), head_num=head_num)
                v_cross = reshape_by_heads(self.Wv_cross(slot_features), head_num=head_num)
                out_concat_cross = multi_head_attention(q_cross, k_cross, v_cross)
                multi_head_out_cross = self.multi_head_combine_cross(out_concat_cross)
                input2 = input2 + multi_head_out_cross
            
            out2 = self.addAndNormalization2(None, input2)
            out2 = self.feedForward(out2)
            out3 = input2 + out2
        
        return out3


# =========================================================================
# MTL MODEL (POMO WITH SLOT DIFFUSION)
# =========================================================================

class SlotDiffModel(nn.Module):
    """
    POMO Model with Slot Diffusion
    
    Features:
    - Pure POMO architecture (no MOE)
    - Slot Attention for representation learning
    - Latent Diffusion for auxiliary task
    - Progressive slot integration
    - Gated slot decoder
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = model_params['eval_type']
        self.problem = model_params['problem']

        # Main components
        self.encoder = MTL_Encoder(**model_params)
        self.decoder = MTL_Decoder(**model_params)
        
        # Slot Diffusion components
        self.enable_diffusion = model_params.get('enable_slot_diffusion', False)
        if self.enable_diffusion:
            feature_dim = 5
            slot_dim = model_params['embedding_dim']
            denoiser_dim = model_params['embedding_dim']
            
            self.denoiser = LDMDenoiser(
                feature_dim=feature_dim,
                slot_dim=slot_dim,
                denoiser_dim=denoiser_dim,
                n_heads=model_params.get('denoiser_heads', 4),
                n_layers=model_params.get('denoiser_layers', 2),
                max_timesteps=model_params.get('max_timesteps', 1000)
            )
            self.max_timesteps = model_params.get('max_timesteps', 1000)
        
        # Legacy reconstruction
        self.enable_reconstruction = model_params.get('enable_slot_reconstruction', False)
        if self.enable_reconstruction:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(model_params['embedding_dim'], 256),
                nn.ReLU(),
                nn.Linear(256, 5)
            )
        
        self.encoded_nodes = None
        self.slots = None
        self.original_features = None
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
        
        # Save for diffusion
        self.original_features = node_xy_demand_tw
        
        # Encode
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_tw)
        
        # Get slots
        self.slots = self.encoder.slot_attention_module.last_slots
        
        # Set decoder KV
        self.decoder.set_kv(self.encoded_nodes, slots=self.slots)

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, selected=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

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
            
            probs = self.decoder(encoded_last_node, attr, ninf_mask=state.ninf_mask)
            
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

        return selected, prob
    
    def compute_slot_diffusion_loss(self):
        """Slot Diffusion Loss"""
        if not self.enable_diffusion or self.slots is None or self.original_features is None:
            return torch.tensor(0.0, device=self.device)
        
        z_0 = self.original_features
        B, N, D = z_0.shape
        
        # Sample timesteps
        t = torch.randint(0, self.max_timesteps, (B,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(z_0)
        
        # Add noise
        alpha_t = 1.0 - t.float() / self.max_timesteps
        alpha_t = alpha_t.view(B, 1, 1)
        z_t = torch.sqrt(alpha_t) * z_0 + torch.sqrt(1 - alpha_t) * noise
        
        # Predict noise
        predicted_noise = self.denoiser(z_t, t, self.slots)
        
        # MSE loss
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        
        return diffusion_loss
    
    def compute_slot_reconstruction_loss(self, reset_state):
        """Legacy Reconstruction Loss"""
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
        """Contrastive Loss for slot diversity"""
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
# ENCODER WITH PROGRESSIVE SLOT INTEGRATION
# =========================================================================

class MTL_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        encoder_layer_num = model_params['encoder_layer_num']
        
        # Feature embedding (Pure Linear, no MOE)
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)
        
        # Slot Attention Module
        self.slot_attention_module = SlotAttentionModule(**model_params)
        
        # Progressive Slot Integration
        self.layers = nn.ModuleList([
            EnhancedEncoderLayer(i, **model_params) for i in range(encoder_layer_num)
        ])

    def forward(self, depot_xy, node_xy_demand_tw):
        # Embed features
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand_tw)
        
        H_nodes = torch.cat((embedded_depot, embedded_node), dim=1)
        
        # Generate slots
        H_slots = self.slot_attention_module(H_nodes)
        
        # Progressive slot integration (layers 3-5)
        out = H_nodes
        for i, layer in enumerate(self.layers):
            slots_to_use = H_slots if i >= len(self.layers) // 2 else None
            out = layer(out, slot_features=slots_to_use)

        return out


# =========================================================================
# DECODER WITH SLOT GATING
# =========================================================================

class MTL_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        
        # Node attention
        self.Wk_nodes = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_nodes = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        # Slot attention
        self.Wk_slots = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_slots = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        # Gating mechanism
        self.slot_gate = nn.Linear(embedding_dim + 4, 1)
        
        # Multi-head combine (Pure Linear, no MOE)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k_nodes = None
        self.v_nodes = None
        self.k_slots = None
        self.v_slots = None
        self.single_head_key = None
        self.slots = None

    def set_kv(self, encoded_nodes, slots=None):
        head_num = self.model_params['head_num']

        self.k_nodes = reshape_by_heads(self.Wk_nodes(encoded_nodes), head_num=head_num)
        self.v_nodes = reshape_by_heads(self.Wv_nodes(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        
        if slots is not None:
            self.slots = slots
            self.k_slots = reshape_by_heads(self.Wk_slots(slots), head_num=head_num)
            self.v_slots = reshape_by_heads(self.Wv_slots(slots), head_num=head_num)

    def forward(self, encoded_last_node, attr, ninf_mask):
        head_num = self.model_params['head_num']

        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)

        # Attention to nodes
        out_concat_nodes = multi_head_attention(q_last, self.k_nodes, self.v_nodes, rank3_ninf_mask=ninf_mask)

        # Attention to slots (if available)
        if self.slots is not None:
            out_concat_slots = multi_head_attention(q_last, self.k_slots, self.v_slots)
            
            # Gating mechanism
            gate_logit = self.slot_gate(input_cat)
            gate_weight = torch.sigmoid(gate_logit)
            
            # Weighted combination
            out_concat = gate_weight * out_concat_slots + (1 - gate_weight) * out_concat_nodes
        else:
            out_concat = out_concat_nodes

        # Multi-head combine
        mh_atten_out = self.multi_head_combine(out_concat)

        # Single-head attention for probability
        score = torch.matmul(mh_atten_out, self.single_head_key)

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