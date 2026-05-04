import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.SlotDiffMOEModel import SlotDiffMOEModel
import envs 

torch.Tensor.cuda = lambda self, *args, **kwargs: self
torch.nn.Module.cuda = lambda self, *args, **kwargs: self
original_cat = torch.cat
torch.cat = lambda tensors, dim=0, **kwargs: original_cat([t.cpu() for t in tensors], dim, **kwargs)

def force_cpu_creation(func):
    def wrapper(*args, **kwargs):
        if 'device' in kwargs: kwargs['device'] = 'cpu'
        return func(*args, **kwargs)
    return wrapper

torch.ones = force_cpu_creation(torch.ones)
torch.zeros = force_cpu_creation(torch.zeros)
torch.tensor = force_cpu_creation(torch.tensor)

def load_model_at_epoch(checkpoint_path, env_params, model_params, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SlotDiffMOEModel(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model

def plot_16_tasks_consistent_instance(epoch, checkpoint_dir, env_params, base_model_params):
    device = torch.device('cpu')
    tasks = [
        'CVRP', 'OVRP', 'VRPB', 'OVRPB', 'VRPL', 'OVRPL',
        'VRPTW', 'OVRPTW', 'VRPBTW', 'OVRPBTW', 'VRPLTW', 'OVRPLTW',
        'VRPBL', 'OVRPBL', 'VRPBLTW', 'OVRPBLTW'
    ]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)
    cmap = plt.get_cmap('hsv')
    
    print(">> Đang chốt Master Instance (Tọa độ vàng)...")
    from envs import OVRPBLTWEnv
    master_env = OVRPBLTWEnv(**env_params)
    master_batch = master_env.get_random_problems(batch_size=1, problem_size=env_params['problem_size'])
    
    # Chỉ bóc đúng 3 lõi: Depot, Tọa độ Nodes, Nhu cầu (Demand)
    m_depot = master_batch[0]
    m_nodes = master_batch[1]
    m_demand = master_batch[2]

    ckpt_path = os.path.join(checkpoint_dir, f"epoch-{epoch}.pt")
    model = load_model_at_epoch(ckpt_path, env_params, base_model_params, device)

    fig, axes = plt.subplots(4, 4, figsize=(24, 24))
    axes = axes.flatten()

    for idx, task_name in enumerate(tasks):
        print(f">> Đang triển khai 'Trojan Horse' vào bài: {task_name}...")
        ax = axes[idx]
        model.problem = task_name
        
        env_class_name = f"{task_name}Env"
        env_class = getattr(envs, env_class_name)
        env = env_class(**env_params)
        
        # BƯỚC 1: Cứ để nó tự sinh ra mâm cỗ chuẩn của nó
        native_batch = env.get_random_problems(batch_size=1, problem_size=env_params['problem_size'])
        
        # BƯỚC 2: Rã đông mâm cỗ thành list để có thể "tráo hàng"
        hacked_batch = list(native_batch)
        
        # BƯỚC 3: Ký sinh Tọa độ và Demand từ Master Instance
        hacked_batch[0] = m_depot
        hacked_batch[1] = m_nodes
        if 'B' in task_name:
            hacked_batch[2] = m_demand # Giữ nguyên demand âm để làm Backhaul
        else:
            hacked_batch[2] = torch.abs(m_demand) # Ép thành dương để không lỗi
            
        # BƯỚC 4: Ép nó ăn cái mâm cỗ đã bị tráo lõi
        env.load_problems(1, tuple(hacked_batch))
        reset_state, _, _ = env.reset()
        
        with torch.no_grad():
            model.pre_forward(reset_state)
            attention = model.encoder.slot_attention_module.last_attention[0].cpu()
            slot_assignments = attention.argmax(dim=0).numpy()
            
            coords = torch.cat((reset_state.depot_xy[0], reset_state.node_xy[0]), dim=0).cpu().numpy()
            raw_demands = reset_state.node_demand[0].cpu().numpy()
            demands = np.abs(raw_demands)
            node_sizes = 30 + demands * 350 
            
            hashed_assignments = (slot_assignments[1:] * 17) % base_model_params['slot_num']
            colors = cmap(hashed_assignments / base_model_params['slot_num'])
            
            # Đánh dấu Vuông/Tròn
            markers = ['o'] * len(coords[1:])
            for i, d in enumerate(raw_demands):
                if d < -1e-5: markers[i] = 's'
            
            for i in range(len(coords[1:])):
                ax.scatter(coords[i+1, 0], coords[i+1, 1], color=colors[i], marker=markers[i],
                           s=node_sizes[i], alpha=0.9, edgecolors='white', linewidth=1.5, zorder=4)
                
                # Hiển thị TW (Lấy chính cái TW native của Env đó sinh ra)
                if 'TW' in task_name and hasattr(reset_state, 'node_tw_start'):
                    tw_start = reset_state.node_tw_start[0, i].item()
                    if i % 4 == 0: 
                        ax.text(coords[i+1, 0]+0.015, coords[i+1, 1]+0.015, f"{tw_start:.1f}", 
                                fontsize=8, color='gray')

            # Vẽ Depot & Mạng nhện
            ax.scatter(coords[0, 0], coords[0, 1], c='black', marker='*', s=400, edgecolors='white', zorder=5)
            unique_slots = np.unique(slot_assignments[1:])
            for slot_id in unique_slots:
                points_in_slot = coords[1:][slot_assignments[1:] == slot_id]
                hash_id = (slot_id * 17) % base_model_params['slot_num']
                color = cmap(hash_id / base_model_params['slot_num'])
                if len(points_in_slot) >= 2:
                    centroid = points_in_slot.mean(axis=0)
                    ax.scatter(centroid[0], centroid[1], color=color, marker='X', s=120, edgecolors='black', zorder=3)
                    for pt in points_in_slot:
                        ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]], color=color, linestyle='--', lw=1.2, alpha=0.3, zorder=2)
                elif len(points_in_slot) == 1:
                    ax.scatter(points_in_slot[0, 0], points_in_slot[0, 1], facecolors='none', edgecolors=color, s=node_sizes[slot_assignments[1:] == slot_id] + 80, lw=2, zorder=2)

            ax.set_title(task_name, fontsize=20, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.15) 
    save_name = f"consistent_instance_16_tasks_epoch5000_size_100.pdf"
    plt.savefig(save_name, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n>> ĐÃ ĐÁNH TRÁO THÀNH CÔNG KHÔNG RỚT 1 BÀI NÀO! FILE TẠI: {save_name}")

if __name__ == "__main__":
    env_params = {'problem_size': 100, 'pomo_size': 100}
    base_model_params = {
        'embedding_dim': 128, 'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6, 'decoder_layer_num': 1,
        'qkv_dim': 16, 'head_num': 8, 'logit_clipping': 10,
        'ff_hidden_dim': 512, 'eval_type': 'argmax',
        'norm': 'instance', 'norm_loc': 'norm_last', 'problem': 'CVRP',
        'num_experts': 4, 'topk': 2, 
        'expert_loc': ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'], 
        'routing_level': 'node', 'routing_method': 'input_choice',
        'slot_num': 32, 'slot_iter_num': 3,
        'enable_slot_diffusion': True, 'max_timesteps': 1000,
        'denoiser_heads': 4, 'denoiser_layers': 2,
        'enable_slot_reconstruction': False,
    } 
    checkpoint_dir = "./checkpoints_100" 
    plot_16_tasks_consistent_instance(5000, checkpoint_dir, env_params, base_model_params)