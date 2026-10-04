# Drive upload bundles

## Current

| Folder | Vai trò |
|---|---|
| [AnchorFlow_v11_NODE](AnchorFlow_v11_NODE/README.md) | Baseline nghiên cứu chính; fresh40 + early stop; true solver-controlled NODE |
| [AnchorFlow_v10_1_LiteMetric](AnchorFlow_v10_1_LiteMetric/README.md) | Control learned-h Euler2, nhanh hơn; dùng cùng data/loss |

Upload nguyên folder đầy đủ vào **MyDrive**, không chỉ notebook/model.py. Notebook bên trong tự kiểm tra checksum. [Paired notebook](../notebooks/AnchorFlow_Compare_V10_1_V11_Fresh40.ipynb) đặt cạnh hai folder trên; đọc [hướng dẫn](../docs/AnchorFlow_V10_1_V11_Benchmark.md).

Kết quả/insight mới nhất: [V11 technical baseline](../docs/AnchorFlow_V11.md). Spec/verification bên trong folder là **frozen release snapshot** trước benchmark; không chỉnh chúng chỉ để đổi wording rồi làm checksum/resume hỏng.

## Historical

`AnchorFlow_Research`, `AnchorFlow_v5_Piecewise`, `AnchorFlow_v6_Connection`, `AnchorFlow_v7_Jet`, `AnchorFlow_v8_Dynamics`, `AnchorFlow_v9_Consensus`, `AnchorFlow_v9_1_MetricRefine`, `AnchorFlow_v10_AdaptiveJet` được giữ để tái lập. Không coi notebook historical là flow train mới nhất; xem [documentation index](../docs/README.md).

Git giữ code/config/doc/notebook và raw metric nhỏ của bundle, không giữ ZIP/checkpoint/teacher weights/data. `.gitattributes` bảo toàn exact bytes để clone trên Windows/Linux không phá bundle SHA256.

Các historical warm-start bundle có thể cần parent checkpoint ngoài Git (ví dụ V9.1 `parent_v9_best_weights.pth`); phải lấy lại asset đúng SHA từ Drive/original ZIP trước khi dùng. Hai current fresh bundles V10.1/V11 không cần parent checkpoint. Manifest V9.1 đã được reseal từ proof/runtime fix có sẵn khi publish, không đổi model weights/forward trong lần này.
