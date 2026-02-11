import torch
import numpy as np
import random
import os
from envs.CVRPEnv import CVRPEnv  # Import class môi trường của bạn

def seed_everything(seed):
    """Khóa seed chặt chẽ"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">> [System] Seed set to: {seed}")

def check_data_generation(problem_size=50, batch_size=5):
    # 1. Khởi tạo môi trường
    # Lưu ý: Các tham số khác không quan trọng, chỉ cần problem_size
    env_params = {
        'problem_size': problem_size, 
        'pomo_size': problem_size, 
        'loc_scaler': None, 
        'device': 'cpu' # Chạy CPU cho nhanh và tiện
    }
    
    # 2. Sinh dữ liệu
    env = CVRPEnv(**env_params)
    depot_xy, node_xy, node_demand = env.get_random_problems(batch_size, problem_size)

    # 3. In ra kết quả kiểm tra
    print(f"\n--- KIỂM TRA DỮ LIỆU SINH RA (Seed 2023) ---")
    print(f"Problem Size: {problem_size}, Batch Size: {batch_size}")
    
    print("\n[Sample 0] Depot Coordinates:")
    print(depot_xy[0]) # In toạ độ kho của bài đầu tiên
    
    print("\n[Sample 0] First 3 Nodes Coordinates:")
    print(node_xy[0][:3]) # In toạ độ 3 node đầu của bài đầu tiên
    
    print("\n[Sample 0] First 3 Nodes Demand:")
    print(node_demand[0][:3]) # In nhu cầu

    # 4. Checksum (Tổng kiểm tra) - Cách nhanh nhất để so sánh
    # Nếu số này giống nhau ở mọi máy -> Dữ liệu giống nhau 100%
    checksum = depot_xy.sum() + node_xy.sum() + node_demand.sum()
    print(f"\n>>> CHECKSUM TOÀN BỘ BATCH: {checksum.item():.6f}")
    print("------------------------------------------------")

if __name__ == "__main__":
    # Test lần 1
    seed_everything(2023)
    check_data_generation()
    
    # Test lần 2 (Chạy lại để chứng minh nó ra y hệt)
    print("\n\n>>> CHẠY LẠI LẦN 2 (Để verify)...")
    seed_everything(2023)
    check_data_generation()