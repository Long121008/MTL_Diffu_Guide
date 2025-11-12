import os, random, math, time
import pytz
import argparse
import pprint as pp
from datetime import datetime

from Trainer import Trainer
from utils import *


def args2dict(args):
    env_params = {"problem_size": args.problem_size, "pomo_size": args.pomo_size}
    
    model_params = {
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
        "slot_num": args.slot_num, 
        "slot_iter_num": args.slot_iter_num,
        
        # NEW: Diffusion Guidance Parameters
        "use_diffusion_guidance": args.use_diffusion_guidance,
        "diffusion_timesteps": args.diffusion_timesteps,
        
        # NEW: Slot Attention Losses
        "enable_slot_reconstruction": args.enable_slot_reconstruction,
    }
    
    optimizer_params = {
        "optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
        "scheduler": {"milestones": args.milestones, "gamma": args.gamma}
    }
    
    trainer_params = {
        "epochs": args.epochs, 
        "train_episodes": args.train_episodes,
        "train_batch_size": args.train_batch_size, 
        "validation_interval": args.validation_interval,
        "model_save_interval": args.model_save_interval, 
        "checkpoint": args.checkpoint,
        
        # NEW: Loss Weights for REINFORCE Training
        "lambda_recon": args.lambda_recon,
        "lambda_contrast": args.lambda_contrast,
        "lambda_guidance": args.lambda_guidance,
    }

    return env_params, model_params, optimizer_params, trainer_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts + Diffusion Guidance")
    
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
    parser.add_argument('--model_type', type=str, default="MOE_LIGHT", 
                       choices=["SINGLE", "MTL", "MOE", "MOE_LIGHT"])
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
    parser.add_argument('--num_experts', type=int, default=4, 
                       help="the number of FFN in a MOE layer")
    parser.add_argument('--eval_type', type=str, default="argmax", 
                       choices=["argmax", "softmax"])
    parser.add_argument('--norm', type=str, default="instance", 
                       choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--norm_loc', type=str, default="norm_last", 
                       choices=["norm_first", "norm_last"], 
                       help="whether conduct normalization before MHA/FFN/MOE")
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
    
    # Slot Attention Parameters
    parser.add_argument('--slot_num', type=int, default=16,
                       help="number of slots for slot attention")
    parser.add_argument('--slot_iter_num', type=int, default=3,
                       help="number of iterations for slot attention")
    parser.add_argument('--enable_slot_reconstruction', action='store_true',
                       help="enable slot reconstruction loss")
    
    # =========================================================================
    # DIFFUSION GUIDANCE PARAMETERS (NEW)
    # =========================================================================
    parser.add_argument('--use_diffusion_guidance', action='store_true',
                       help="enable diffusion guidance for policy improvement")
    parser.add_argument('--diffusion_timesteps', type=int, default=50,
                       help="number of diffusion timesteps (T in DDPM)")
    
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
    parser.add_argument('--epochs', type=int, default=100, 
                       help="total training epochs")
    parser.add_argument('--train_episodes', type=int, default=10000 * 2, 
                       help="the num. of training instances per epoch")
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--validation_interval', type=int, default=100)
    parser.add_argument('--model_save_interval', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help="resume training")
    
    # =========================================================================
    # LOSS WEIGHTS (NEW)
    # =========================================================================
    parser.add_argument('--lambda_recon', type=float, default=0.1,
                       help="weight for slot reconstruction loss")
    parser.add_argument('--lambda_contrast', type=float, default=0.05,
                       help="weight for slot contrastive loss")
    parser.add_argument('--lambda_guidance', type=float, default=0.3,
                       help="weight for diffusion guidance loss (only if use_diffusion_guidance=True)")

    # =========================================================================
    # SETTINGS (e.g., GPU)
    # =========================================================================
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--log_dir', type=str, default="./results")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--occ_gpu', type=float, default=0., 
                       help="occupy (X)% GPU memory in advance, please use sparingly.")

    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    pp.pprint(vars(args))
    print("=" * 80)
    
    # Validation checks
    if args.problem == "ALL" and args.model_type == "Single":
        raise ValueError("Cannot solve multiple problems with Single model, please use MTL/MOE/MOE_LIGHT instead.")
    
    if args.use_diffusion_guidance:
        print("\n" + "=" * 80)
        print("🔥 DIFFUSION GUIDANCE ENABLED 🔥")
        print("=" * 80)
        print(f"  - Diffusion timesteps: {args.diffusion_timesteps}")
        print(f"  - Guidance weight (lambda): {args.lambda_guidance}")
        print(f"  - Slot reconstruction: {args.enable_slot_reconstruction}")
        print(f"  - Recon weight: {args.lambda_recon}")
        print(f"  - Contrast weight: {args.lambda_contrast}")
        print("=" * 80 + "\n")
    
    env_params, model_params, optimizer_params, trainer_params = args2dict(args)
    seed_everything(args.seed)

    # Set log & gpu
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

    # Start training
    print("\n" + "=" * 80)
    print(">> Start {} Training using {} Model ...".format(args.problem, args.model_type))
    if args.use_diffusion_guidance:
        print(">> Training Mode: REINFORCE + Diffusion Guidance")
    else:
        print(">> Training Mode: REINFORCE (Baseline)")
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