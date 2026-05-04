import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_individual_task_scores(csv_path='train_log.csv', bin_size=50):
    print(f">> Đang đọc dữ liệu từ: {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] Lỗi: Không tìm thấy file {csv_path}.")
        return

    task_score_cols = ['score_CVRP', 'score_OVRP', 'score_OVRPTW', 'score_VRPB', 'score_VRPL', 'score_VRPTW']
    
    # Kỹ thuật làm mượt: Lấy trung bình mỗi block 50 epochs
    print(f">> Đang làm mượt dữ liệu (Trung bình mỗi {bin_size} Epochs)...")
    df['epoch_bin'] = (df['epoch'] // bin_size) * bin_size
    df_sampled = df.groupby('epoch_bin').mean().reset_index()

    # Cài đặt style cho đồ thị chuẩn Paper
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2.5
    })

    # Lấy 1 bộ 6 màu đẹp đẹp để mỗi bài 1 màu cho rực rỡ
    colors = sns.color_palette("husl", len(task_score_cols))

    print(">> Đang xuất xưởng 6 Đồ thị ra 6 file PDF riêng biệt...")
    for idx, col in enumerate(task_score_cols):
        task_name = col.replace('score_', '')
        
        plt.figure(figsize=(7, 4.5)) # Thu nhỏ size lại chút xíu vì chỉ vẽ 1 đường
        
        # Vẽ đường cho task hiện tại
        plt.plot(df_sampled['epoch_bin'], df_sampled[col], color=colors[idx], label=task_name)
        
        plt.title(f'Score Convergence: {task_name}')
        plt.xlabel('Training Epoch')
        plt.ylabel('Score ')
        
        # Thêm lưới nét đứt cho dễ nhìn
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Lưu file riêng biệt, ví dụ: train_score_CVRP.pdf
        save_name = f'train_score_{task_name}.pdf'
        plt.savefig(save_name, format='pdf', dpi=300)
        plt.close()
        
        print(f"   -> Đã xuất: {save_name}")

    print("\n=========================================================")
    print("🎉 XONG!")
    print("=========================================================")

if __name__ == "__main__":
    CSV_FILE_PATH = "train_log.csv"
    SMOOTHING_WINDOW = 100 
    
    plot_individual_task_scores(csv_path=CSV_FILE_PATH, bin_size=SMOOTHING_WINDOW)