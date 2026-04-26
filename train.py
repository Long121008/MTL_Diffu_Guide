import os, random, math, time
import pytz
import argparse
import pprint as pp
from datetime import datetime

from Trainer import Trainer
from utils import *


def args2dict(args):
    env_params = {
        "problem_size": args.problem_size, 
        "pomo_size": args.pomo_size
    }
    
    model_params = {
        # Architecture
        "embedding_dim": args.embedding_dim, 
        "sqrt_embedding_dim": args.sqrt_embedding_dim,
        "encoder_layer_num": args.encoder_layer_num, 
        "decoder_layer_num": args.decoder_layer_num,
        "qkv_dim": args.qkv_dim, 
        "head_num": args.head_num, 
        "logit_clipping": args.logit_clipping,
        "ff_hidden_dim": args.ff_hidden_dim, 
        "num_experts": args.num_experts, 
        "eval_type": args.eval_type,
        "norm": args.norm, 
        "norm_loc": args.norm_loc, 
        "expert_loc": args.expert_loc, 
        "problem": args.problem,
        "topk": args.topk, 
        "routing_level": args.routing_level, 
        "routing_method": args.routing_method,
        
        # Slot Attention
        "slot_num": args.slot_num, 
        "slot_iter_num": args.slot_iter_num,
        
        # ========================================
        # DUAL-TASK: Slot Diffusion (NEW!)
        # ========================================
        "enable_slot_diffusion": args.enable_slot_diffusion,
        "max_timesteps": args.max_timesteps,
        "denoiser_heads": args.denoiser_heads,
        "denoiser_layers": args.denoiser_layers,
        
        # Legacy Reconstruction (optional)
        "enable_slot_reconstruction": args.enable_slot_reconstruction,
    }
    
    optimizer_params = {
        "optimizer": {
            "lr": args.lr, 
            "weight_decay": args.weight_decay
        },
        "scheduler": {
            "milestones": args.milestones, 
            "gamma": args.gamma
        }
    }
    
    trainer_params = {
        "epochs": args.epochs, 
        "train_episodes": args.train_episodes,
        "train_batch_size": args.train_batch_size, 
        "validation_interval": args.validation_interval,
        "model_save_interval": args.model_save_interval, 
        "checkpoint": args.checkpoint,
        
        # ========================================
        # DUAL-TASK: Loss Weights (NEW!)
        # ========================================
        "lambda_diffusion": args.lambda_diffusion,
        "lambda_recon": args.lambda_recon,
        "lambda_contrastive": args.lambda_contrastive,
    }

    return env_params, model_params, optimizer_params, trainer_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts + Slot Diffusion")
    
    # =========================================================================
    # ENV PARAMS
    # =========================================================================
    parser.add_argument('--problem', type=str, default="Train_ALL", 
                        choices=["Train_ALL", "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
                                "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
                                "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"])
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--pomo_size', type=int, default=50, 
                        help="the number of start node, should <= problem size")

    # =========================================================================
    # MODEL PARAMS
    # =========================================================================
    parser.add_argument('--model_type', type=str, default="SlotDiff", choices=["MOE_LIGHT_Mixed", "MOE_Mixed", "MTL_Mixed","SINGLE", "MTL", "MOE", "MOE_LIGHT", "SlotDiffModel", "SlotModel", "SlotDiffMOEModel"])
    
    # Basic Architecture
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=float, default=128**(1/2))
    parser.add_argument('--encoder_layer_num', type=int, default=6, 
                        help="the number of MHA in encoder")
    parser.add_argument('--decoder_layer_num', type=int, default=1, 
                        help="the number of MHA in decoder")
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=float, default=10)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    
    # MoE Settings
    parser.add_argument('--num_experts', type=int, default=4, 
                        help="the number of FFN in a MOE layer")
    parser.add_argument('--topk', type=int, default=2, 
                        help="how many experts (on average) to route for each input")
    parser.add_argument('--expert_loc', type=str, nargs='+', 
                        default=['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'], 
                        help="where to use MOE")
    parser.add_argument('--routing_level', type=str, default="node", 
                        choices=["node", "instance", "problem"], 
                        help="routing level for MOE")
    parser.add_argument('--routing_method', type=str, default="input_choice", 
                        choices=["input_choice", "expert_choice", "soft_moe", "random"], 
                        help="only for token-level and instance-level routing")
    
    # Evaluation & Normalization
    parser.add_argument('--eval_type', type=str, default="argmax", 
                        choices=["argmax", "softmax"])
    parser.add_argument('--norm', type=str, default="instance", 
                        choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--norm_loc', type=str, default="norm_last", 
                        choices=["norm_first", "norm_last"], 
                        help="whether conduct normalization before MHA/FFN/MOE")
    
    # Slot Attention
    parser.add_argument('--slot_num', type=int, default=16,
                        help="number of slots for slot attention")
    parser.add_argument('--slot_iter_num', type=int, default=3,
                        help="number of iterations for slot attention")
    
    # =========================================================================
    # DUAL-TASK: SLOT DIFFUSION (NEW!)
    # =========================================================================
    parser.add_argument('--enable_slot_diffusion', action='store_true', default=False,
                        help="Enable slot diffusion for representation learning")
    parser.add_argument('--max_timesteps', type=int, default=1000,
                        help="Maximum timesteps for diffusion process")
    parser.add_argument('--denoiser_heads', type=int, default=4,
                        help="Number of attention heads in denoiser")
    parser.add_argument('--denoiser_layers', type=int, default=2,
                        help="Number of transformer layers in denoiser")
    
    # Legacy Reconstruction (optional, usually disabled)
    parser.add_argument('--enable_slot_reconstruction', action='store_true', default=False,
                        help="Enable legacy slot reconstruction loss")

    # =========================================================================
    # OPTIMIZER PARAMS
    # =========================================================================
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--milestones', type=int, nargs='+', default=[4501, ], 
                        help='when to decay lr')
    parser.add_argument('--gamma', type=float, default=0.1, 
                        help='new_lr = lr * gamma')

    # =========================================================================
    # TRAINER PARAMS
    # =========================================================================
    parser.add_argument('--epochs', type=int, default=5000, 
                        help="total training epochs")
    parser.add_argument('--train_episodes', type=int, default=10000 * 2, 
                        help="the num. of training instances per epoch")
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--validation_interval', type=int, default=5000)
    parser.add_argument('--model_save_interval', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help="resume training")
    
    # =========================================================================
    # DUAL-TASK: LOSS WEIGHTS (NEW!)
    # =========================================================================
    parser.add_argument('--lambda_diffusion', type=float, default=0.1,
                        help="Weight for slot diffusion loss (Nhánh 2: Representation Learning)")
    parser.add_argument('--lambda_recon', type=float, default=0.0,
                        help="Weight for legacy reconstruction loss (usually 0)")
    parser.add_argument('--lambda_contrastive', type=float, default=0.01,
                        help="Weight for slot contrastive loss (ensure slot diversity)")

    # =========================================================================
    # SETTINGS (GPU, SEED, etc.)
    # =========================================================================
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--log_dir', type=str, default="./results")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--occ_gpu', type=float, default=0., 
                        help="occupy (X)% GPU memory in advance, please use sparingly.")

    # =========================================================================
    # PARSE & VALIDATE
    # =========================================================================
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("CONFIGURATION:")
    print("=" * 80)
    pp.pprint(vars(args))
    print("=" * 80)
    
    # Convert args to dict
    env_params, model_params, optimizer_params, trainer_params = args2dict(args)
    
    # Seed everything
    seed_everything(args.seed)

    # Validation
    if args.problem == "ALL" and args.model_type == "Single":
        assert False, "Cannot solve multiple problems with Single model, please use MTL/MOE/MOE_LIGHT instead."
    
    if args.enable_slot_diffusion:
        print("\n>> DUAL-TASK MODE ENABLED:")
        print(f"   - Nhánh 1 (Policy): VRP Solver")
        print(f"   - Nhánh 2 (Representation): Slot Diffusion")
        print(f"   - Diffusion Timesteps: {args.max_timesteps}")
        print(f"   - Denoiser: {args.denoiser_layers} layers, {args.denoiser_heads} heads")
        print(f"   - Loss Weights: λ_diffusion={args.lambda_diffusion}, λ_contrastive={args.lambda_contrastive}")
        print()

    # =========================================================================
    # SETUP LOGGING & GPU
    # =========================================================================
    torch.set_printoptions(threshold=1000000)
    process_start_time = datetime.now(pytz.timezone("Asia/Singapore"))
    args.log_path = os.path.join(args.log_dir, process_start_time.strftime("%Y%m%d_%H%M%S"))
    print(">> Log Path: {}".format(args.log_path))
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    if not args.no_cuda and torch.cuda.is_available():
        occumpy_mem(args) if args.occ_gpu != 0. else print(">> No occupation needed")
        args.device = torch.device('cuda', args.gpu_id)
        torch.cuda.set_device(args.gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    
    print(">> USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(not args.no_cuda, args.gpu_id))

    # =========================================================================
    # START TRAINING
    # =========================================================================
    print("\n" + "=" * 80)
    print(">> Start {} Training using {} Model ...".format(args.problem, args.model_type))
    print("=" * 80 + "\n")
    
    trainer = Trainer(
        args=args, 
        env_params=env_params, 
        model_params=model_params, 
        optimizer_params=optimizer_params, 
        trainer_params=trainer_params
    )
    trainer.run()
    
    print("\n" + "=" * 80)
    print(">> Finish {} Training using {} Model ...".format(args.problem, args.model_type))
    print("=" * 80)