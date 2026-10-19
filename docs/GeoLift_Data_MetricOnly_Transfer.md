# Chuyển GeoLift_Data sang tài khoản Colab khác — metric-only

**Chỉ cần metric coarse về TEACHER, không phải chỉ cần một file dữ liệu.**
S4-U-PIR NoRelative và DecoderCapacity đều cần RGB/sparse depth/GT/calibration
để train và validation; metric teacher là supervision bổ sung ở train.

Giữ nguyên cấu trúc trên tài khoản mới:

```text
MyDrive/GeoLift_Data/
  teacher_subset_2000/
    selected_2000_ids.json
    kitti_trainval_2000.tar
    metric_coarse_train_2000.tar
  test_1000/
    kitti_test_1000.tar
```

| File | Mục đích | Cần mang? |
|---|---|---|
| `selected_2000_ids.json` | Khóa đúng1600train/400val và IDs | **Bắt buộc** |
| `kitti_trainval_2000.tar` | RGB, sparse, GT, K và split files | **Bắt buộc** |
| `metric_coarse_train_2000.tar` | Metric teacher cache đúng protocol | **Bắt buộc khi teacher ON** |
| `kitti_test_1000.tar` |1000 ảnh test + sparse/K cho prediction benchmark | Nên mang; thiếu thì `prepare` tải KITTI official và tạo lại |

Không mang cho các run này: `relative_teacher*.tar`,
`relative_DA3MONO_LARGE_generation/`, relative reports/previews,
`anchorflow_cache/da2_small_*`, teacher model weights, các candidate manifests cũ.
Không thay metric coarse bằng DA2/relative cache: code kiểm tra hash/nội dung.
Không cần chuyển run/checkpoint cũ để **fresh train** G0/H/I; backup chúng riêng
nếu muốn resume run cũ hoặc export Orin.

## Đã kiểm tra bản local19/10/2026

`D:\GeoLift_Data` chứa10 ZIP Google Drive export:001,005–013; chưa có002–004.
Inventory của các ZIP đang có tìm được split JSON, trainval TAR và test1000 TAR,
**chưa tìm thấy `metric_coarse_train_2000.tar`**. Không kết luận file đó chắc chắn
ở phần ZIP nào; cần tải đúng file từ Drive cũ hoặc bổ sung export còn thiếu.

Folder lọc nằm ở:

`D:\GeoLift_MetricOnly_Transfer\GeoLift_Data`

Script chỉ copy3file tìm được, kiểm tra CRC của ZIP và SHA256 đọc lại output;
split SHA phải là `f0f482799db63b1e89f4ef21be0aff71cc409bac8bde20259192f3bc1cd40ddb`.
Các ZIP gốc/relative cache không bị xóa. Trạng thái thiếu file và hash từng file
ghi trong `D:\GeoLift_MetricOnly_Transfer\transfer_manifest.json`.
**Đây chưa phải bộ train đủ cho đến khi bổ sung metric coarse TAR.**

Sau khi có file thiếu, chép vào `teacher_subset_2000/`, rồi upload **folder con
`GeoLift_Data`** vào MyDrive tài khoản mới. Không upload folder wrapper
`GeoLift_MetricOnly_Transfer` thay cho `GeoLift_Data`. Mount Drive bằng tài khoản
mới; đường dẫn `/content/drive/MyDrive/GeoLift_Data` giữ nguyên, không phải sửa
email/account trong config. Upload riêng folder code/notebook train mới.

## Cell kiểm tra ở Colab mới

```python
from pathlib import Path
import hashlib
root = Path('/content/drive/MyDrive/GeoLift_Data')
required = [
    'teacher_subset_2000/selected_2000_ids.json',
    'teacher_subset_2000/kitti_trainval_2000.tar',
    'teacher_subset_2000/metric_coarse_train_2000.tar',
]
for name in required:
    path = root/name
    print(name, 'OK' if path.is_file() else 'MISSING')
assert all((root/name).is_file() for name in required), 'Bổ sung file còn thiếu trước khi train'
assert hashlib.sha256((root/required[0]).read_bytes()).hexdigest() == \
    'f0f482799db63b1e89f4ef21be0aff71cc409bac8bde20259192f3bc1cd40ddb'
print('Test1000:', (root/'test_1000/kitti_test_1000.tar').is_file())
```

Vẫn chạy `prepare` của notebook để audit nội dung teacher, paired IDs và leakage;
cell trên chỉ kiểm tra file/split, không thay thế data gates.
