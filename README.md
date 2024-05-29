# LSTMSignLanguage
Đồ án môn học: Học máy

Dataset: Data for Dynamic Vietnamese Sign Language (http://test101.udn.vn/d-VSL/) (Truy cập lần cuối 29/05/2024)

Label: Cảm ơn, Cha, Chào, Mẹ, Ông, Tạm biệt.

Video được đưa vào Mediapipe Holistic để rút trích các landmarks trên cơ thể, mỗi landmarks lấy 2 tọa độ điểm x, y. Đưa các đặc trưng này vào mô hình LSTM để học các đặc trưng.
