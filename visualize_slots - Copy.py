import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import model và env của bro
from models.SlotDiffMOEModel import SlotDiffMOEModel
from envs import CVRPEnv  

# Vô hiệu hóa khả năng đẩy lên cuda của mọi Tensor
torch.Tensor.cuda = lambda self, *args, **kwargs: self
torch.nn.Module.cuda = lambda self, *args, **kwargs: self

# Ép các hàm khởi tạo luôn dùng CPU
original_cat = torch.cat
torch.cat = lambda tensors, dim=0, **kwargs: original_cat([t.cpu() for t in tensors], dim, **kwargs)

# Chống lỗi dòng 341 cụ thể: Ép mọi tensor mới tạo ra về CPU
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

def plot_slot_evolution_spiderweb(epochs_to_plot, checkpoint_dir, data_batch, env_params, model_params):
    device = torch.device('cpu')
    num_epochs = len(epochs_to_plot)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    
    fig, axes = plt.subplots(1, num_epochs, figsize=(8 * num_epochs, 8))
    if num_epochs == 1:
        axes = [axes]
        
    # ĐỔI BẢNG MÀU SANG 'hsv' ĐỂ CÓ DẢI MÀU LIÊN TỤC TRÁNH ĐỤNG HÀNG
    cmap = plt.get_cmap('hsv')

    for idx, epoch in enumerate(epochs_to_plot):
        print(f">> Processing Epoch {epoch}...")
        ckpt_path = os.path.join(checkpoint_dir, f"epoch-{epoch}.pt")
        model = load_model_at_epoch(ckpt_path, env_params, model_params, device)
        
        with torch.no_grad():
            model.pre_forward(data_batch)
            
            # Lấy instance đầu tiên
            attention = model.encoder.slot_attention_module.last_attention[0].cpu()
            slot_assignments = attention.argmax(dim=0).numpy()
            
            # Tọa độ: Depot ở index 0, Nodes ở index 1->N
            coords = torch.cat((data_batch.depot_xy[0], data_batch.node_xy[0]), dim=0).cpu().numpy()
            
            # Lấy demand của các node khách hàng (shape: [N])
            demands = data_batch.node_demand[0].cpu().numpy()
            # Scale kích thước điểm vẽ (demand càng lớn, node càng to)
            node_sizes = 30 + demands * 350 
            
            ax = axes[idx]
            
            # 1. Vẽ các điểm (Nodes) với Bubble Chart
            # CẬP NHẬT: normalize array c cho khớp với bảng màu hsv (chia cho slot_num)
            ax.scatter(coords[1:, 0], coords[1:, 1], 
                       c=slot_assignments[1:] / model_params['slot_num'], cmap=cmap, 
                       s=node_sizes, alpha=0.9, edgecolors='white', linewidth=1.5, zorder=4)
            
            # Đánh dấu Depot bằng hình ngôi sao lớn
            ax.scatter(coords[0, 0], coords[0, 1], c='black', marker='*', s=400, edgecolors='white', zorder=5, label="Depot")
            
            # 2. Vẽ Mạng nhện (Spider Web) nối các node về trọng tâm của Slot
            unique_slots = np.unique(slot_assignments[1:]) # Bỏ qua depot
            for slot_id in unique_slots:
                points_in_slot = coords[1:][slot_assignments[1:] == slot_id]
                
                # CẬP NHẬT: Map màu chính xác theo ID của slot chia cho tổng số slot
                color = cmap(slot_id / model_params['slot_num'])
                
                if len(points_in_slot) >= 2:
                    # Tìm trọng tâm (centroid) của Slot
                    centroid = points_in_slot.mean(axis=0)
                    
                    # Vẽ cái tâm bằng dấu X
                    ax.scatter(centroid[0], centroid[1], color=color, marker='X', s=120, edgecolors='black', linewidth=1, zorder=3)
                    
                    # Nối từng node về tâm bằng nét đứt
                    for pt in points_in_slot:
                        ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]], color=color, linestyle='--', lw=1.5, alpha=0.5, zorder=2)
                        
                elif len(points_in_slot) == 1:
                    # Nếu Slot chỉ có 1 node, chỉ khoanh viền đậm lên cho dễ nhận biết
                    ax.scatter(points_in_slot[0, 0], points_in_slot[0, 1], facecolors='none', edgecolors=color, s=node_sizes[slot_assignments[1:] == slot_id] + 50, lw=2, zorder=2)

            ax.set_title(f"Slot Centroid & Demand Analysis (Epoch {epoch})", fontsize=20, fontweight='bold', pad=15)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Viền ngoài đồ thị
            for spine in ax.spines.values():
                spine.set_edgecolor('#CCCCCC')
                spine.set_linewidth(2)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1) 
    plt.savefig("slot_evolution_spiderweb.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(">> Done! Saved to slot_evolution_spiderweb.pdf")
    plt.show()

if __name__ == "__main__":
    # 1. Định nghĩa params
    env_params = {
        'problem_size': 50, 
        'pomo_size': 50
    }
    
    model_params = {
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
        'problem': 'CVRP', 
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
    
    # Khởi tạo Env và data
    env = CVRPEnv(**env_params)
    data_batch = env.get_random_problems(batch_size=1, problem_size=50) 
    env.load_problems(1, data_batch)
    reset_state, _, _ = env.reset()
    
    # Cho vào list các epoch bro muốn vẽ
    epochs_to_plot = [5000]
    checkpoint_dir = "./checkpoints" 
    
    plot_slot_evolution_spiderweb(epochs_to_plot, checkpoint_dir, reset_state, env_params, model_params)