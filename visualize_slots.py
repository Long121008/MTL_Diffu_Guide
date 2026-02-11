import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from scipy.spatial import ConvexHull
from utils import get_model
from models.SlotDiffModel import SlotDiffModel # Hoặc class mô hình của bạn

def visualize_slots_fixed(checkpoint_path, model_type="SlotDiffModel", problem_size=50, num_plots=4, seed=2024):
    """
    Vẽ biểu đồ Slot với dữ liệu CỐ ĐỊNH (để so sánh giữa các epoch).
    """
    # 1. KHÓA SEED (QUAN TRỌNG NHẤT)
    print(f">> Đang khóa Random Seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")

    # 3. Load Model Params (Copy từ train.py)
    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**0.5,
        'encoder_layer_num': 6,
        'decoder_layer_num': 1,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'norm': 'instance',
        'norm_loc': 'norm_last',
        'num_experts': 4,
        'topk': 2,
        'expert_loc': [],
        'routing_level': 'node',
        'routing_method': 'input_choice',
        'problem': 'CVRP',
        
        # Slot & Diffusion
        'slot_num': 16,
        'slot_iter_num': 3,
        'enable_slot_diffusion': True,
        'enable_slot_reconstruction': False,
        'denoiser_heads': 4,
        'denoiser_layers': 2,
        'max_timesteps': 1000,
    }

    print(">> Khởi tạo Model...")
    if model_type == "SlotDiffModel":
        model = SlotDiffModel(**model_params).to(device)
    # else: model = SlotDiffMOEModel(**model_params).to(device)

    # 4. Load Checkpoint
    print(f">> Loading checkpoint: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print(f"ERROR: Không tìm thấy file {checkpoint_path}")
        return

    model.eval()

    # 5. Tạo dữ liệu (Sẽ LUÔN GIỐNG NHAU nhờ seed)
    batch_size = num_plots
    
    # Depot: [Batch, 1, 2] (Đã fix dimension)
    depot_xy = torch.rand(batch_size, 1, 2).to(device)
    
    # Node: [Batch, N, 2]
    node_xy = torch.rand(batch_size, problem_size, 2).to(device)
    
    # Dummy features
    node_demand = torch.rand(batch_size, problem_size).to(device)
    node_tw_start = torch.zeros(batch_size, problem_size).to(device)
    node_tw_end = torch.ones(batch_size, problem_size).to(device)

    class DummyState:
        def __init__(self):
            self.depot_xy = depot_xy
            self.node_xy = node_xy
            self.node_demand = node_demand
            self.node_tw_start = node_tw_start
            self.node_tw_end = node_tw_end
    
    reset_state = DummyState()

    # 6. Inference
    print(">> Đang chạy Inference...")
    with torch.no_grad():
        model.pre_forward(reset_state)

    # 7. Lấy Attention & Data
    attn_weights = model.encoder.slot_attention_module.last_attention.cpu().numpy()
    node_xy = node_xy.cpu().numpy()
    depot_xy = depot_xy.cpu().numpy()

    # 8. Vẽ hình
    print(">> Đang vẽ hình...")
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1: axes = [axes]
    cmap = plt.get_cmap('tab20')

    for b in range(num_plots):
        ax = axes[b]
        coords = node_xy[b] # [N, 2]
        depot = depot_xy[b][0] # [2] (Lấy phần tử 0)
        
        # Lấy Slot ID
        attn_b = attn_weights[b] # [Slots, N+1]
        attn_nodes = attn_b[:, 1:] # [Slots, N]
        node_slot_ids = np.argmax(attn_nodes, axis=0) 
        
        # Vẽ Depot
        ax.scatter(depot[0], depot[1], c='black', marker='*', s=300, label='Depot', zorder=20)
        
        # Vẽ các cụm
        unique_slots = np.unique(node_slot_ids)
        for slot_id in unique_slots:
            mask = (node_slot_ids == slot_id)
            points = coords[mask]
            color = cmap(slot_id / 20)
            
            # Vẽ điểm
            ax.scatter(points[:, 0], points[:, 1], c=[color], s=80, edgecolors='k', zorder=10)
            
            # Vẽ vùng bao (Convex Hull) - Có try/except
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    poly = patches.Polygon(hull_points, closed=True, facecolor=color, alpha=0.2, edgecolor=color, linewidth=2, zorder=5)
                    ax.add_patch(poly)
                except Exception:
                    pass 
            
            # Vẽ nhãn S_id
            if len(points) > 0:
                center = points.mean(axis=0)
                ax.text(center[0], center[1], f"S{slot_id}", fontsize=12, fontweight='bold', 
                        color='black', ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=30)

        ax.set_title(f"Sample {b+1}", fontsize=14)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    # Lưu file với tên Epoch (tự động lấy từ tên file checkpoint)
    epoch_name = os.path.basename(checkpoint_path).replace(".pt", "")
    save_name = f"viz_{epoch_name}_seed{seed}.png"
    
    plt.savefig(save_name, dpi=300)
    print(f">> Đã lưu ảnh: {save_name}")
    # plt.show() # Tắt show nếu chạy trên server không màn hình

if __name__ == "__main__":
    # CÁCH DÙNG:
    # Chạy lần 1: Visualize Epoch 500
    ##visualize_slots_fixed(CKPT_1, model_type="SlotDiffModel", seed=1234) # <--- Seed cố định
    
    # Chạy lần 2: Visualize Epoch 1000
    #CKPT_2 = "./pretrained/SlotDIff/epoch-1000.pt"
    #visualize_slots_fixed(CKPT_2, model_type="SlotDiffModel", seed=1234) # <--- Seed GIỐNG HỆT

     CKPT_4 = "./pretrained/SlotDIff/epoch-4000.pt"
     visualize_slots_fixed(CKPT_4, model_type="SlotDiffModel", seed=1234) # <--- Seed GIỐNG HỆT