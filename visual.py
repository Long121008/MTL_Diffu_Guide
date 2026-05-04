import os
import glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from models.SlotDiffMOEModel import SlotDiffMOEModel
import envs 

# ==========================================
# HACK ÉP CPU (ĐỂ KHÔNG BỊ LỖI BỘ NHỚ CUDA)
# ==========================================
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

def load_model_at_epoch(checkpoint_path, model_params, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SlotDiffMOEModel(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model

# ==========================================
# PARSER HELPER FUNCTIONS
# ==========================================
def parse_cvrplib(file_path):
    with open(file_path, "r") as file:
        lines = [ll.strip() for ll in file]
    i, dimension, capacity = 0, 0, 1
    locations, demand = None, None
    
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"): dimension = int(line.split(':')[1])
        elif line.startswith("CAPACITY"): capacity = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i += dimension
        elif line.startswith('DEMAND_SECTION'):
            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i += dimension
        i += 1
        
    original_locations = locations[:, 1:]
    original_locations = np.expand_dims(original_locations, axis=0)
    loc_scaler = 1000
    locations = original_locations / loc_scaler
    
    depot_xy = torch.Tensor(locations[:, :1, :])
    node_xy = torch.Tensor(locations[:, 1:, :])
    node_demand = torch.Tensor(demand[1:, 1:].reshape((1, -1))) / capacity 

    problem_size = node_xy.size(1)
    env_params = {'problem_size': problem_size, 'pomo_size': problem_size, 'loc_scaler': loc_scaler}
    env = envs.CVRPEnv(**env_params)
    env.load_problems(1, (depot_xy, node_xy, node_demand))
    reset_state, _, _ = env.reset()
    return reset_state, env_params, 'CVRP'

def parse_solomon(file_path):
    with open(file_path, "r") as file:
        lines = [ll.strip() for ll in file]
    i, capacity, data = 0, 1, None
    while i < len(lines):
        line = lines[i]
        if line.startswith("NUMBER"):
            capacity = int(lines[i+1].split(' ')[-1])
        elif line.startswith("CUST NO."):
            data_lines = [lines[j] for j in range(i+1, len(lines)) if lines[j] != '']
            data = np.loadtxt(data_lines, dtype=int)
            break
        i += 1
        
    original_locations = data[:, 1:3]
    original_locations = np.expand_dims(original_locations, axis=0) 
    scaler = max(original_locations.max(), data[0, 5] / 3.) 
    locations = original_locations / scaler
    
    depot_xy = torch.Tensor(locations[:, :1, :])
    node_xy = torch.Tensor(locations[:, 1:, :])
    node_demand = torch.Tensor(data[1:, 3].reshape((1, -1))) / capacity
    service_time = torch.Tensor(data[1:, -1].reshape((1, -1))) / scaler 
    tw_start = torch.Tensor(data[1:, 4].reshape((1, -1))) / scaler 
    tw_end = torch.Tensor(data[1:, 5].reshape((1, -1))) / scaler 

    problem_size = node_xy.size(1)
    env_params = {'problem_size': problem_size, 'pomo_size': problem_size, 'loc_scaler': scaler}
    env = envs.VRPTWEnv(**env_params)
    env.depot_end = data[0, 5] / scaler
    env.load_problems(1, (depot_xy, node_xy, node_demand, service_time, tw_start, tw_end))
    reset_state, _, _ = env.reset()
    return reset_state, env_params, 'VRPTW'

# ==========================================
# MASTER BATCH PLOTTER 
# ==========================================
def plot_all_to_pdf(folder_paths, output_pdf_name, epoch, checkpoint_dir, base_model_params):
    device = torch.device('cpu')
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)
    cmap = plt.get_cmap('hsv')

    # Lấy danh sách tất cả các file
    all_files = []
    for folder in folder_paths:
        # Lấy file CVRPLIB
        all_files.extend(glob.glob(os.path.join(folder, '*.vrp')))
        # Lấy file Solomon
        all_files.extend(glob.glob(os.path.join(folder, '*.txt')))
        
    total_files = len(all_files)
    if total_files == 0:
        print(">> Không tìm thấy file nào trong các thư mục chú chỉ định!")
        return
        
    print(f">> Tìm thấy tổng cộng {total_files} bài toán. Đang bắt đầu quá trình in ấn...")

    # Load model ban đầu (sẽ thay đổi tham số 'problem' bên trong vòng lặp)
    ckpt_path = os.path.join(checkpoint_dir, f"epoch-{epoch}.pt")
    model = load_model_at_epoch(ckpt_path, base_model_params, device)

    # Khởi tạo file PDF nhiều trang
    with PdfPages(output_pdf_name) as pdf:
        fig, axes = None, None
        
        for idx, file_path in enumerate(all_files):
            instance_name = os.path.basename(file_path).split('.')[0]
            print(f">> Đang xử lý [{idx+1}/{total_files}]: {instance_name}...")
            
            # Cứ mỗi 16 hình thì tạo một Figure mới
            if idx % 16 == 0:
                fig, axes = plt.subplots(4, 4, figsize=(24, 24))
                axes = axes.flatten()
                
            ax = axes[idx % 16]
            
            # 1. Phân loại và lấy dữ liệu
            try:
                if file_path.endswith('.vrp'):
                    reset_state, env_params, task_name = parse_cvrplib(file_path)
                elif file_path.endswith('.txt'):
                    reset_state, env_params, task_name = parse_solomon(file_path)
                else:
                    continue
            except Exception as e:
                print(f"   -> Lỗi đọc file {instance_name}: {e}")
                ax.set_title(f"{instance_name}\n(Error Reading)", color='red')
                ax.axis('off')
                continue

            # 2. Forward model
            model.problem = task_name # Đổi problem để route đúng chuyên gia MoE
            with torch.no_grad():
                model.pre_forward(reset_state)
                attention = model.encoder.slot_attention_module.last_attention[0].cpu()
                slot_assignments = attention.argmax(dim=0).numpy()
                
                coords = torch.cat((reset_state.depot_xy[0], reset_state.node_xy[0]), dim=0).cpu().numpy()
                demands = np.abs(reset_state.node_demand[0].cpu().numpy())
                node_sizes = 30 + demands * 350 
                
                hashed_assignments = (slot_assignments[1:] * 17) % base_model_params['slot_num']
                colors = cmap(hashed_assignments / base_model_params['slot_num'])
                
                # 3. Vẽ lên ax hiện tại
                ax.scatter(coords[1:, 0], coords[1:, 1], color=colors, marker='o',
                           s=node_sizes, alpha=0.9, edgecolors='white', linewidth=1.5, zorder=4)
                
                if task_name == 'VRPTW' and hasattr(reset_state, 'node_tw_start'):
                    scaler = env_params['loc_scaler']
                    for i in range(len(coords[1:])):
                        if i % 3 == 0: # In thưa cho đỡ rối
                            tw_s = reset_state.node_tw_start[0, i].item() * scaler
                            tw_e = reset_state.node_tw_end[0, i].item() * scaler
                            ax.text(coords[i+1, 0]+0.015, coords[i+1, 1]+0.015, f"[{tw_s:.0f}-{tw_e:.0f}]", 
                                    fontsize=7, color='gray', zorder=6)

                ax.scatter(coords[0, 0], coords[0, 1], c='black', marker='*', s=400, edgecolors='white', zorder=5)
                
                unique_slots = np.unique(slot_assignments[1:])
                for slot_id in unique_slots:
                    points_in_slot = coords[1:][slot_assignments[1:] == slot_id]
                    hash_id = (slot_id * 17) % base_model_params['slot_num']
                    c = cmap(hash_id / base_model_params['slot_num'])
                    if len(points_in_slot) >= 2:
                        centroid = points_in_slot.mean(axis=0)
                        ax.scatter(centroid[0], centroid[1], color=c, marker='X', s=120, edgecolors='black', zorder=3)
                        for pt in points_in_slot:
                            ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]], color=c, linestyle='--', lw=1.2, alpha=0.4, zorder=2)
                    elif len(points_in_slot) == 1:
                        ax.scatter(points_in_slot[0, 0], points_in_slot[0, 1], facecolors='none', edgecolors=c, s=node_sizes[slot_assignments[1:] == slot_id] + 80, lw=2.5, zorder=2)

                # Format ô
                ax.set_title(f"{instance_name} (Size {env_params['problem_size']})", fontsize=18, fontweight='bold')
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor('#DDDDDD')
                    spine.set_linewidth(1.5)

            # 4. Kiểm tra xem đã vẽ đủ 16 hình hoặc đã đến file cuối cùng chưa
            is_last_figure_of_page = (idx + 1) % 16 == 0
            is_last_file_of_all = (idx == total_files - 1)
            
            if is_last_figure_of_page or is_last_file_of_all:
                # Nếu là trang cuối mà không chẵn 16 ô thì ẩn các ô trắng đi
                if is_last_file_of_all and not is_last_figure_of_page:
                    for j in range((idx % 16) + 1, 16):
                        axes[j].axis('off')
                        
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.1, hspace=0.15) 
                
                # Lưu trang hiện tại vào file PDF
                pdf.savefig(fig)
                plt.close(fig) # Đóng để giải phóng RAM
                print(f" >> Đã lưu xong 1 trang vào PDF!")

    print(f"\n=======================================================")
    print(f"🎉 ĐÃ HOÀN THÀNH XUẤT BẢN CUỐN CATALOGUE: {output_pdf_name} 🎉")
    print(f"=======================================================")

if __name__ == "__main__":
    base_model_params = {
        'embedding_dim': 128, 'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6, 'decoder_layer_num': 1,
        'qkv_dim': 16, 'head_num': 8, 'logit_clipping': 10,
        'ff_hidden_dim': 512, 'eval_type': 'argmax',
        'norm': 'instance', 'norm_loc': 'norm_last', 'problem': 'CVRP', # Tạm thời
        'num_experts': 4, 'topk': 2, 
        'expert_loc': ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'], 
        'routing_level': 'node', 'routing_method': 'input_choice',
        'slot_num': 32, 'slot_iter_num': 3,
        'enable_slot_diffusion': True, 'max_timesteps': 1000,
        'denoiser_heads': 4, 'denoiser_layers': 2,
        'enable_slot_reconstruction': False,
    } 
    checkpoint_dir = "./checkpoints_100" 
    
    # CHÚ ĐIỀN ĐƯỜNG DẪN CÁC THƯ MỤC CẦN QUÉT VÀO ĐÂY NHÉ:
    # Chú để ý cái ảnh chú gửi, điền đúng tên folder thực tế trên máy chú nha.
    folders_to_scan = [
        "data/CVRP-LIB", 
        "data/CVRP-LIB 2", 
        "data/Vrp-Set-Solomon"
    ]
    
    output_filename = "Slot_Attention_Full_Benchmark_Catalogue.pdf"
    
    plot_all_to_pdf(folders_to_scan, output_filename, 5000, checkpoint_dir, base_model_params)