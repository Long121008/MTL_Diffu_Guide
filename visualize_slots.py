import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.SlotDiffMOEModel import SlotDiffMOEModel
import envs  # Giả sử module envs của chú chứa tất cả các class môi trường

# Vô hiệu hóa khả năng đẩy lên cuda của mọi Tensor
torch.Tensor.cuda = lambda self, *args, **kwargs: self
torch.nn.Module.cuda = lambda self, *args, **kwargs: self

original_cat = torch.cat
torch.cat = lambda tensors, dim=0, **kwargs: original_cat([t.cpu() for t in tensors], dim, **kwargs)

def force_cpu_creation(func):
    def wrapper(*args, **kwargs):
        if 'device' in kwargs:
            kwargs['device'] = 'cpu'
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

def plot_16_tasks_appendix(epoch, checkpoint_dir, env_params, base_model_params):
    device = torch.device('cpu')
    
    # Danh sách 16 bài toán theo đúng Table 11
    tasks = [
        'CVRP', 'OVRP', 'VRPB', 'OVRPB', 'VRPL', 'OVRPL',
        'VRPTW', 'OVRPTW', 'VRPBTW', 'OVRPBTW', 'VRPLTW', 'OVRPLTW',
        'VRPBL', 'OVRPBL', 'VRPBLTW', 'OVRPBLTW'
    ]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2) # Chỉnh scale chữ nhỏ lại xíu cho grid 4x4
    cmap = plt.get_cmap('hsv')
    
    # Load model chung
    ckpt_path = os.path.join(checkpoint_dir, f"epoch-{epoch}.pt")
    model = load_model_at_epoch(ckpt_path, env_params, base_model_params, device)

    # KHỞI TẠO GRID 4x4 SIÊU TO KHỔNG LỒ
    fig, axes = plt.subplots(4, 4, figsize=(24, 24))
    axes = axes.flatten() # Ép thành mảng 1 chiều để dễ truy cập bằng index

    for idx, task_name in enumerate(tasks):
        print(f">> Đang vẽ ô {idx+1}/16: {task_name}...")
        
        # Lấy cái trục (axis) tương ứng cho bài toán này
        ax = axes[idx]
        
        # Cập nhật problem trong model_params để MoE route đúng chuyên gia
        model.problem = task_name
        
        # Lấy class môi trường tương ứng
        env_class_name = f"{task_name}Env"
        if not hasattr(envs, env_class_name):
            print(f"Bỏ qua {task_name} vì không tìm thấy class {env_class_name} trong module envs.")
            ax.set_title(f"{task_name} (Not Found)", color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
            
        env_class = getattr(envs, env_class_name)
        env = env_class(**env_params)
        
        data_batch = env.get_random_problems(batch_size=1, problem_size=env_params['problem_size']) 
        env.load_problems(1, data_batch)
        reset_state, _, _ = env.reset()
        
        with torch.no_grad():
            model.pre_forward(reset_state)
            
            attention = model.encoder.slot_attention_module.last_attention[0].cpu()
            slot_assignments = attention.argmax(dim=0).numpy()
            
            coords = torch.cat((reset_state.depot_xy[0], reset_state.node_xy[0]), dim=0).cpu().numpy()
            demands = np.abs(reset_state.node_demand[0].cpu().numpy())
            node_sizes = 30 + demands * 350 
            
            # Xáo trộn màu để các slot tách bạch
            hashed_assignments = (slot_assignments[1:] * 17) % base_model_params['slot_num']
            colors = cmap(hashed_assignments / base_model_params['slot_num'])
            
            # Phân tách Maker: Tròn cho Linehaul, Vuông cho Backhaul (nếu có)
            markers = ['o'] * len(coords[1:])
            if 'B' in task_name and hasattr(reset_state, 'backhaul_mask'):
                bh_mask = reset_state.backhaul_mask[0].cpu().numpy()
                for i, is_bh in enumerate(bh_mask[1:]):
                    if is_bh: markers[i] = 's' # Square cho Backhaul
            
            # Vẽ từng điểm để có thể custom marker
            for i in range(len(coords[1:])):
                ax.scatter(coords[i+1, 0], coords[i+1, 1], 
                           color=colors[i], marker=markers[i],
                           s=node_sizes[i], alpha=0.9, edgecolors='white', linewidth=1.5, zorder=4)
                
                # Thể hiện Time Window nếu có TW
                if 'TW' in task_name and hasattr(reset_state, 'node_tw_start'):
                    tw_start = reset_state.node_tw_start[0, i].item()
                    tw_end = reset_state.node_tw_end[0, i].item()
                    if i % 3 == 0: 
                        ax.text(coords[i+1, 0]+0.015, coords[i+1, 1]+0.015, f"[{tw_start:.1f}-{tw_end:.1f}]", 
                                fontsize=8, color='gray', zorder=6)
            
            # Depot
            ax.scatter(coords[0, 0], coords[0, 1], c='black', marker='*', s=400, edgecolors='white', zorder=5, label="Depot")
            
            # Vẽ Mạng nhện (Spider Web)
            unique_slots = np.unique(slot_assignments[1:])
            for slot_id in unique_slots:
                points_in_slot = coords[1:][slot_assignments[1:] == slot_id]
                hash_id = (slot_id * 17) % base_model_params['slot_num']
                color = cmap(hash_id / base_model_params['slot_num'])
                
                if len(points_in_slot) >= 2:
                    centroid = points_in_slot.mean(axis=0)
                    ax.scatter(centroid[0], centroid[1], color=color, marker='X', s=120, edgecolors='black', linewidth=1, zorder=3)
                    for pt in points_in_slot:
                        ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]], color=color, linestyle='--', lw=1.5, alpha=0.5, zorder=2)
                elif len(points_in_slot) == 1:
                    ax.scatter(points_in_slot[0, 0], points_in_slot[0, 1], facecolors='none', edgecolors=color, s=node_sizes[slot_assignments[1:] == slot_id] + 80, lw=2.5, zorder=2)

            # Format từng ô
            ax.set_title(task_name, fontsize=20, fontweight='bold', pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#CCCCCC')
                spine.set_linewidth(2)

    # SAU KHI VẼ XONG 16 Ô, LƯU LẠI THÀNH 1 FILE DUY NHẤT
    plt.tight_layout()
    # Dãn khoảng cách giữa các ô một chút cho thoáng
    plt.subplots_adjust(wspace=0.1, hspace=0.15) 
    
    save_name = f"appendix_16_tasks_grid_size100.pdf"
    plt.savefig(save_name, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig) 
    print(f"\n>> ĐÃ XONG! Lưu toàn bộ 16 bài toán vào siêu phẩm: {save_name}")

if __name__ == "__main__":
    env_params = {
        'problem_size': 100, 
        'pomo_size': 100
    }
    
    base_model_params = {
        'embedding_dim': 128, 
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6, 
        'decoder_layer_num': 1,
        'qkv_dim': 16, 
        'head_num': 8, 
        'logit_clipping': 10,
        'ff_hidden_dim': 512, 
        'eval_type': 'argmax',
        'norm': 'instance', 
        'norm_loc': 'norm_last', 
        'problem': 'CVRP', # Kẻ thế mạng lúc khởi tạo
        'num_experts': 4, 
        'topk': 2, 
        'expert_loc': ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'], 
        'routing_level': 'node', 
        'routing_method': 'input_choice',
        'slot_num': 32,
        'slot_iter_num': 3,
        'enable_slot_diffusion': True,
        'max_timesteps': 1000,
        'denoiser_heads': 4,
        'denoiser_layers': 2,
        'enable_slot_reconstruction': False,
    } 
    
    checkpoint_dir = "./checkpoints" 
    
    # Chạy phát xả ra luôn 1 grid 4x4
    plot_16_tasks_appendix(5000, checkpoint_dir, env_params, base_model_params)