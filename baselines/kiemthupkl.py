import pickle
import pprint

# Tên tệp tin của bạn
file_path = 'c:/Users/HC COMPUTER/Prj_MTL/Routing-MVMoE/data_train3phase/CVRP/cvrp50_uniform.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print("Đã tải dữ liệu thành công. Nội dung tệp tin là:")
    # Sử dụng pprint để in dữ liệu một cách dễ đọc hơn
    pprint.pprint(data)

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp {file_path}")
except pickle.UnpicklingError:
    print("Lỗi: Không thể đọc tệp pickle. Tệp có thể bị hỏng hoặc không đúng định dạng.")
except Exception as e:
    print(f"Một lỗi khác đã xảy ra: {e}")

