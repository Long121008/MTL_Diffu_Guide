## MTL Diffu Guide

This repository contains neural solvers for multiple Vehicle Routing Problem (VRP) variants, with a focus on a slot-enhanced Mixture-of-Experts policy model:

- `SlotDiffMOEModel`: MoE policy + Slot Attention + optional Slot Diffusion auxiliary objective
- Additional baselines and ablations in `models/` and `baselines/`

The main training and evaluation entry points are:

- `train.py`
- `test.py`

## 1. Model Architecture (SlotDiffMOEModel)

Implementation: `models/SlotDiffMOEModel.py`

### 1.1 High-level pipeline

1. Input features per customer node:
	 - `(x, y, demand, tw_start, tw_end)`
	 - Depot uses `(x, y)`
2. `MTL_Encoder` embeds depot + nodes into latent tokens.
3. `SlotAttentionModule` extracts `K` latent slots (global structure tokens).
4. Encoder layers process node tokens, and deeper layers use slot cross-attention.
5. `MTL_Decoder` performs a dual-pointer decision:
	 - Node pointer (local state driven)
	 - Slot pointer (global structure driven)
	 - Score-level gated fusion chooses the next node probabilities.
6. Training uses REINFORCE (POMO rollout) + optional auxiliary losses.

### 1.2 Core components

#### A) Slot Attention branch

- `SlotAttentionModule` initializes trainable slot priors (`mu`, `log_sigma`), samples slots, then refines them with `IterativeAttention` for `slot_iter_num` steps.
- The final attention map and slots are cached for auxiliary objectives and analysis.

#### B) Enhanced encoder with slot integration

- `EnhancedEncoderLayer` has standard self-attention.
- In later encoder layers, it also performs cross-attention from node tokens to slot tokens.
- Feed-forward block can be either:
	- Standard MLP (`FeedForward`), or
	- MoE (`MoE`) depending on `num_experts` and `expert_loc`.

#### C) Dual-pointer decoder (key idea)

`MTL_Decoder` computes two score tensors over candidate nodes:

- `score_nodes`: from node-context pointer (local decision signal)
- `score_slots`: from slot-context pointer (global structural signal)

Then it uses a capped gate:

- `gate = 0.3 * sigmoid(slot_gate(...))`
- Final score (before mask):

	`score = gate * score_slots + (1 - gate) * score_nodes`

Masking is applied once after fusion to avoid numerical issues with `-inf` combinations.

### 1.3 Training objectives

In `Trainer.py`, total loss is:

`L_total = L_RL + lambda_diffusion * L_diffusion + lambda_recon * L_recon + lambda_contrastive * L_contrastive + L_aux`

Where:

- `L_RL`: REINFORCE loss from POMO rollout
- `L_diffusion`: denoising loss from `LDMDenoiser` on node feature noise prediction
- `L_recon`: optional legacy slot reconstruction loss
- `L_contrastive`: slot diversity regularization
- `L_aux`: auxiliary MoE load-balancing loss (if model exposes `aux_loss`)

If `--enable_slot_diffusion` is not set, diffusion branch is disabled.

## 2. Supported problem modes

From `utils.get_env`:

- Training multi-task set: `Train_ALL` = `CVRP, OVRP, VRPB, VRPL, VRPTW, OVRPTW`
- Testing all set: `ALL` includes 16 VRP variants.

## 3. Setup

### 3.1 Environment

Use Python 3.10+ (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note:

- `requirements.txt` pins `torch==2.7.1+cu121` (CUDA build).
- If your machine is CPU-only or uses another CUDA version, install a compatible Torch build first, then install remaining packages.

### 3.2 Data

Datasets are expected under `data/` (already present in this workspace).

## 4. How to Train

### 4.1 Recommended SlotDiff-MoE training command

```bash
python train.py \
	--model_type SlotDiffMOEModel \
	--problem Train_ALL \
	--problem_size 50 \
	--pomo_size 50 \
	--embedding_dim 128 \
	--encoder_layer_num 6 \
	--qkv_dim 16 \
	--head_num 8 \
	--ff_hidden_dim 512 \
	--num_experts 4 \
	--topk 2 \
	--expert_loc Enc0 Enc1 Enc2 Enc3 Enc4 Enc5 Dec \
	--routing_level node \
	--routing_method input_choice \
	--slot_num 16 \
	--slot_iter_num 3 \
	--enable_slot_diffusion \
	--max_timesteps 1000 \
	--denoiser_heads 4 \
	--denoiser_layers 2 \
	--lambda_diffusion 0.1 \
	--lambda_contrastive 0.01 \
	--epochs 5000 \
	--train_episodes 20000 \
	--train_batch_size 128 \
	--gpu_id 0
```

Outputs:

- Logs/checkpoints are created under `results/<timestamp>/`
- Model checkpoints are saved as `epoch-<N>.pt`
- Training metrics are saved to `train_log.csv`

### 4.2 Resume training

```bash
python train.py \
	--model_type SlotDiffMOEModel \
	--problem Train_ALL \
	--checkpoint results/<timestamp>/epoch-1000.pt
```

## 5. How to Test

### 5.1 Evaluate checkpoint on all problem variants

```bash
python test.py \
	--model_type SlotDiffMOEModel \
	--problem ALL \
	--problem_size 50 \
	--pomo_size 50 \
	--checkpoint results/<timestamp>/epoch-5000.pt \
	--slot_num 32 \
	--slot_iter_num 3 \
	--enable_slot_diffusion \
	--max_timesteps 1000 \
	--denoiser_heads 4 \
	--denoiser_layers 2 \
	--test_episodes 1000 \
	--test_batch_size 100 \
	--aug_factor 8 \
	--gpu_id 0
```

### 5.2 Evaluate one problem only

```bash
python test.py \
	--model_type SlotDiffMOEModel \
	--problem CVRP \
	--problem_size 50 \
	--checkpoint results/<timestamp>/epoch-5000.pt \
	--slot_num 32 \
	--slot_iter_num 3 \
	--enable_slot_diffusion
```

### 5.3 Evaluate benchmark instance folder/file

You can point `--test_set_path` to:

- a directory of `.vrp` files (CVRPLIB style), or
- a directory/file of `.txt` benchmark instances (e.g., Solomon VRPTW).

Example:

```bash
python test.py \
	--model_type SlotDiffMOEModel \
	--problem CVRP \
	--checkpoint results/<timestamp>/epoch-5000.pt \
	--test_set_path data/CVRP-LIB \
	--slot_num 32 \
	--slot_iter_num 3 \
	--enable_slot_diffusion
```

Test outputs:

- CSV summaries are written to `results/test/<timestamp>/test_results_<PROBLEM>.csv`

## 6. Important configuration compatibility rules

When running `test.py`, keep architecture flags consistent with the checkpoint:

- `--model_type`
- `--embedding_dim`, `--encoder_layer_num`, `--qkv_dim`, `--head_num`, `--ff_hidden_dim`
- MoE settings (`--num_experts`, `--topk`, `--expert_loc`, routing)
- Slot settings (`--slot_num`, `--slot_iter_num`)
- Diffusion settings (`--enable_slot_diffusion`, `--max_timesteps`, `--denoiser_heads`, `--denoiser_layers`)

Mismatch can cause state-dict loading errors or degraded quality.

## 7. Quick file map

- `models/SlotDiffMOEModel.py`: SlotDiff-MoE architecture
- `Trainer.py`: training loop and total loss composition
- `Tester.py`: evaluation loop, benchmark solvers, CSV export
- `utils.py`: model/env factory and helper utilities
- `data/`: VRP datasets and benchmark instances

## 8. Minimal quickstart

```bash
# Train
python train.py --model_type SlotDiffMOEModel --problem Train_ALL --enable_slot_diffusion

# Test
python test.py --model_type SlotDiffMOEModel --problem ALL --checkpoint results/<timestamp>/epoch-5000.pt --enable_slot_diffusion
```

