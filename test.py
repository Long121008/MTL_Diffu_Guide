import os, random, math, time
import pytz
import argparse
import pprint as pp
from datetime import datetime

from Tester import Tester
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
        "problem": None,  # Để None vì test đa bài toán
        "topk": args.topk, 
        "routing_level": args.routing_level, 
        "routing_method": args.routing_method,
        
        # ========================================
        # Slot Attention (CRITICAL!)
        # ========================================
        "slot_num": args.slot_num,
        "slot_iter_num": args.slot_iter_num,
        
        # ========================================
        # Dual-Task: Slot Diffusion
        # ========================================
        "enable_slot_diffusion": args.enable_slot_diffusion,
        "max_timesteps": args.max_timesteps,
        "denoiser_heads": args.denoiser_heads,
        "denoiser_layers": args.denoiser_layers,
        
        # Legacy Reconstruction
        "enable_slot_reconstruction": args.enable_slot_reconstruction,
    }
    
    tester_params = {
        "checkpoint": args.checkpoint, 
        "test_episodes": args.test_episodes, 
        "test_batch_size": args.test_batch_size,
        "sample_size": args.sample_size, 
        "aug_factor": args.aug_factor, 
        "aug_batch_size": args.aug_batch_size,
        "test_set_path": args.test_set_path, 
        "test_set_opt_sol_path": args.test_set_opt_sol_path,
        "fine_tune_episodes": args.fine_tune_episodes, 
        "fine_tune_epochs": args.fine_tune_epochs,
        "fine_tune_batch_size": args.fine_tune_batch_size, 
        "fine_tune_aug_factor": args.fine_tune_aug_factor,
        "lr": args.lr, 
        "weight_decay": args.weight_decay
    }

    return env_params, model_params, tester_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts + Slot Diffusion - TESTING")
    
    # =========================================================================
    # ENV PARAMS
    # =========================================================================
    parser.add_argument('--problem', type=str, default="ALL", 
                        choices=["ALL", "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
                                "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
                                "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"])
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--pomo_size', type=int, default=50, 
                        help="the number of start node, should <= problem size")

    # =========================================================================
    # MODEL PARAMS (MUST MATCH TRAINING CONFIG!)
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
                        help="how many ffn(s) to route for each input")
    parser.add_argument('--expert_loc', type=str, nargs='+', 
                        default=['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'], 
                        help="where to use MOE layer")
    parser.add_argument('--routing_level', type=str, default="node", 
                        choices=["node", "instance", "problem"], 
                        help="routing level for MOE")
    parser.add_argument('--routing_method', type=str, default="input_choice", 
                        choices=["input_choice", "expert_choice", "soft_moe", "random"], 
                        help="only activate for instance-level and token-level routing")
    
    # Evaluation & Normalization
    parser.add_argument('--eval_type', type=str, default="argmax", 
                        choices=["argmax", "softmax"])
    parser.add_argument('--norm', type=str, default="instance", 
                        choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--norm_loc', type=str, default="norm_last", 
                        choices=["norm_first", "norm_last"], 
                        help="whether conduct normalization before MHA/FFN/MOE")
    
    # =========================================================================
    # SLOT ATTENTION (CRITICAL - MUST MATCH TRAINING!)
    # =========================================================================
    parser.add_argument('--slot_num', type=int, default=16,
                        help="number of slots for slot attention (must match training)")
    parser.add_argument('--slot_iter_num', type=int, default=3,
                        help="number of iterations for slot attention (must match training)")
    
    # =========================================================================
    # DUAL-TASK: SLOT DIFFUSION (MUST MATCH TRAINING!)
    # =========================================================================
    parser.add_argument('--enable_slot_diffusion', action='store_true', default=False,
                        help="Enable slot diffusion (set if trained with diffusion)")
    parser.add_argument('--max_timesteps', type=int, default=1000,
                        help="Maximum timesteps for diffusion (must match training)")
    parser.add_argument('--denoiser_heads', type=int, default=4,
                        help="Number of attention heads in denoiser (must match training)")
    parser.add_argument('--denoiser_layers', type=int, default=2,
                        help="Number of transformer layers in denoiser (must match training)")
    
    # Legacy Reconstruction
    parser.add_argument('--enable_slot_reconstruction', action='store_true', default=False,
                        help="Enable legacy slot reconstruction (set if trained with reconstruction)")

    # =========================================================================
    # TESTER PARAMS
    # =========================================================================
    parser.add_argument('--checkpoint', type=str, default="./checkpoint/epoch-100.pt", 
                        help="load pretrained model to evaluate")
    parser.add_argument('--test_episodes', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--sample_size', type=int, default=10, 
                        help="only activate if eval_type is softmax")
    parser.add_argument('--aug_factor', type=int, default=8, choices=[1, 8], 
                        help="whether to use instance augmentation during evaluation")
    parser.add_argument('--aug_batch_size', type=int, default=100)
    parser.add_argument('--test_set_path', type=str, default=None, 
                        help="evaluate on default test dataset if None")
    parser.add_argument('--test_set_opt_sol_path', type=str, default=None, 
                        help="evaluate on default test dataset if None")

    # Fine-tuning (optional)
    parser.add_argument('--fine_tune_epochs', type=int, default=0, 
                        help="fine tune the pretrained model if > 0")
    parser.add_argument('--fine_tune_episodes', type=int, default=10000)
    parser.add_argument('--fine_tune_batch_size', type=int, default=64 * 2)
    parser.add_argument('--fine_tune_aug_factor', type=int, default=1, choices=[1, 8], 
                        help="whether to use instance augmentation during fine tuning")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    # =========================================================================
    # SETTINGS (GPU, SEED, etc.)
    # =========================================================================
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--occ_gpu', type=float, default=0., 
                        help="occumpy (X)% GPU memory in advance, please use sparingly.")

    # =========================================================================
    # PARSE & SETUP
    # =========================================================================
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEST CONFIGURATION:")
    print("=" * 80)
    pp.pprint(vars(args))
    print("=" * 80)
    
    env_params, model_params, tester_params = args2dict(args)
    seed_everything(args.seed)

    if args.aug_factor != 1:
        args.test_batch_size = args.aug_batch_size
        tester_params['test_batch_size'] = tester_params['aug_batch_size']

    # =========================================================================
    # SETUP GPU
    # =========================================================================
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
    # VALIDATE CHECKPOINT
    # =========================================================================
    if not os.path.exists(args.checkpoint):
        print(f"\n{'='*80}")
        print(f"ERROR: Checkpoint not found at: {args.checkpoint}")
        print(f"{'='*80}\n")
        exit(1)
    
    # Load checkpoint to check configuration
    print(f"\n>> Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu',weights_only=False)
    
    if 'epoch' in checkpoint:
        print(f">> Checkpoint Info:")
        print(f"   - Epoch: {checkpoint['epoch']}")
        if 'problem' in checkpoint:
            print(f"   - Problem: {checkpoint['problem']}")
    
    # Warn if model architecture might mismatch
    if args.enable_slot_diffusion:
        print(f"\n>> Testing with Dual-Task Model:")
        print(f"   - Slot Attention: {args.slot_num} slots, {args.slot_iter_num} iterations")
        print(f"   - Slot Diffusion: ENABLED")
        print(f"   - Denoiser: {args.denoiser_layers} layers, {args.denoiser_heads} heads")
        print(f"   ⚠️  Make sure checkpoint was trained with these settings!")
    else:
        print(f"\n>> Testing with Standard Model:")
        print(f"   - Slot Attention: {args.slot_num} slots, {args.slot_iter_num} iterations")
        print(f"   - Slot Diffusion: DISABLED")

    # =========================================================================
    # START TESTING
    # =========================================================================
    print("\n" + "=" * 80)
    print(">> Start {} Testing ...".format(args.problem))
    print("=" * 80 + "\n")
    
    tester = Tester(
        args=args, 
        env_params=env_params, 
        model_params=model_params, 
        tester_params=tester_params
    )
    tester.run()
    
    print("\n" + "=" * 80)
    print(">> Finish {} Testing ...".format(args.problem))
    print("=" * 80)