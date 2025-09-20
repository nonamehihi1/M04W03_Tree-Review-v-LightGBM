# M04W03_Tree-Review-v-LightGBM
Tree Review và LightGBM


I. Tree review
Buổi hôm T3 chỉ là ôn tập lại tất cả những loại cây đã học

Loss Function for Regression:
<img width="847" height="349" alt="image" src="https://github.com/user-attachments/assets/76443650-5b02-419d-b8b2-77e04f40e1af" />

Loss Function for Classification:
 <img width="898" height="382" alt="image" src="https://github.com/user-attachments/assets/495371cd-9cab-4a3b-bd45-7e0ea1123207" />

Tree Building:
<img width="890" height="336" alt="image" src="https://github.com/user-attachments/assets/0c114c53-72f9-444e-887c-3ce74dfefb5f" />

II. Light GBM
LightGBM là một framework cho thuật toán Gradient Boosting Decision Tree (GBDT).
Điểm mạnh:
+ Rất nhanh nhờ kỹ thuật tối ưu hóa như:
+ Histogram-based algorithm (tạo histogram để chia node nhanh hơn).
+ Leaf-wise growth thay vì level-wise như XGBoost.
+ Ít tốn bộ nhớ, có thể huấn luyện với dataset rất lớn.
+ Hỗ trợ parallel learning, GPU learning.
+ Xử lý tốt cả dữ liệu categorical mà không cần phải one-hot encode.


Histogram-based: 
1. Chọn số lượng bin(thùng) B
2. Tính toán chiều rộng của thùng = (max - min)/ B
3. Sau đó tạo các ngưỡng mới

Ví dụ: có dữ liệu 1 2 3 4 5, Các chọn ngưỡng bình thường sẽ có 4 ngưỡng : [1.5, 2.5, 3.5, 4.5]. Áp dụng histogram vào, chọn B = 2 ==> chiều rộng = (5-1)/ 2 = 2 ==> ngưỡng mới sẽ là [1, 3, 5]  


Leaf-wise Growth:
Sự khác nhau giữa Level_wise và Leaf_wise là gì?
+ Level-wise: chia từng tầng một, tất cả các node cùng 1 tầng sẽ cùng chia đồng thời
+ Leaf-wise : chỉ chọn lá có Gain lớn nhất để 


Exckusive Feature Bundling: 
Trong nhiều bài toán (ví dụ: one-hot encoding..), có hàng ngàn đặc trưng, nếu lưu trữ và tính toán cho từng đặc trưng riêng biệt --> rất tốn bộ nhớ

Ý tưởng là Nếu 2 hoặc nhiều đặc trưng hầu như không bao giờ khác 0 cùng lúc, ta có thể gộp chúng lại thành 1 feature duy nhất

Ví dụ: có 3 màu Red, Green, Blue theo cách thông thường thì ta sẽ đặt lần lượt là 0, 1, 2. Nhưng sẽ có bật cập, nếu chia ngưỡng theo cách lấy trung bình thì ta sẽ có 2 ngưỡng là 0.5 và 1.5. Khi đó sẽ có 2 trường hợp, nếu ngưỡng là 0.5 thì bên trái là 0(Red) và bên phải là 1 2(Green, Blue), tương tự nếu ngưỡng là 1.5 thì bên trái là 0,1 (Red, Green) và bên phải là 2(Blue), như vậy ta sẽ không thể suy luận ra nhánh nào có màu là Green được. Lý do là bởi vì DT coi giá trị 0,1,2 là có thứ tự, nên việc gán số vô tình làm model nghĩ rằng Red>Green>Blue. Ta có thể hiểu đơn giản rằng các con số 0,1,2 ở đây là các số nguyên, không phải là các số liên tục.

Ta sẽ sử dụng One_hot_encoding:
| Sample | Red | Green | Blue |
| ------ | --- | ---- | ----- |
| 0      | 1   | 0    | 0     |
| 1      | 0   | 1    | 0     |
| 2      | 0   | 0    | 1     |

Ta có thể gộp từ 3 hàng thành 1 hàng 'sample' để tiết kiệm dung lượng bộ nhớ. 


Gradient-based One-Side Sampling (GOSS):
GOSS chọn mẫu thông minh hơn dựa trên gradient:
+ Những sample có |gradient| lớn → đang bị dự đoán sai nhiều → quan trọng cho việc cập nhật mô hình.
+ Những sample có |gradient| nhỏ → mô hình đã dự đoán gần đúng → ít quan trọng.
Vậy nên:
+ Giữ lại toàn bộ các sample có |gradient| lớn.
+ Lấy mẫu ngẫu nhiên (random) một phần nhỏ các sample có |gradient| nhỏ.
Nhưng vì bạn bỏ đi nhiều sample gradient nhỏ, bạn cần tăng trọng số (weight) cho phần gradient nhỏ được giữ lại → để không bị thiên lệch khi tính gain.

Ví dụ: Giả sử bạn dự đoán điểm thi cho 5 học sinh:
| HS | Dự đoán | Điểm thật | Gradient (≈ y - p) |
| -- | ------- | --------- | ------------------ |
| A  | 6       | 9         | +3 (lớn)           |
| B  | 8       | 9         | +1 (nhỏ)           |
| C  | 3       | 8         | +5 (rất lớn)       |
| D  | 5       | 5         | 0 (gần đúng)       |
| E  | 4       | 3         | -1 (nhỏ)           |

Ta thấy: 
+ C và A có gradient lớn --> giữ chắc chắn
+ B và E có gradient nhỏ --> chọn ngẫu nhiên(có thể giữ 1 trong 2)
+ D ~ 0 --> có thể bỏ
