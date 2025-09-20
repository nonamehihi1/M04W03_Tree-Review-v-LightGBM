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
