# Google Drive Dataset Download Guide for Colab

This guide outlines the most efficient methods for pulling datasets from Google Drive into a Google Colab environment, avoiding common I/O bottlenecks and download limits.

> Final fused-geometry notebook: [`../notebooks/GeoLift_RT_v2_1_Geometry_Inspect_Final_Train.ipynb`](../notebooks/GeoLift_RT_v2_1_Geometry_Inspect_Final_Train.ipynb). Baseline metric+DA notebook: [`../notebooks/GeoLift_RT_v2_1_Colab_Train_Test.ipynb`](../notebooks/GeoLift_RT_v2_1_Colab_Train_Test.ipynb).
>
> The two public archives below are **train-split teacher caches**, not the KITTI RGB/sparse/GT dataset. `metric_coarse_train.tar` supplies metric KD and `depth_anything_train_raw.tar` supplies DA2 relative geometry. Full RSGD additionally requires a DSINE cache.

## GeoLift-S2 v2.1: nguồn dữ liệu đã chốt

| Nguồn | Dung lượng hiển thị trên Drive | Nội dung dùng |
|---|---:|---|
| `1ZtRVY67l3QkdgegSxDYtf6Cq_l-_BjQW` | 27 GB | `metric_coarse_train.tar`, metric teacher |
| `1DOtv8E_zW-pOkms2XUqro9vVfMnkqzbB` | 14 GB | `depth_anything_train_raw.tar`, DA relative teacher |
| `1gcaq8rFOOTEUBiGxF05ZBzomi1n2AX1q` | 32.28 GB | `geometry_fused_train.tar`, fused `R_G/C_G` teacher |
| [KITTI Depth Completion](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php) | theo archive chính thức | RGB, sparse LiDAR, GT train/val và 1.000 test anonymous |

Notebook [`GeoLift_RT_v2_1_Colab_Train_Test.ipynb`](../notebooks/GeoLift_RT_v2_1_Colab_Train_Test.ipynb) không ghép tùy ý KITTI với teacher. Nó lấy giao ID giữa hai teacher cache, chọn 1.000 mẫu seed cố định thành 800 train + 200 validation, rồi trích đúng KITTI frame tương ứng. `test_1000` được lấy từ `data_depth_selection.zip` chính thức.

Notebook final lấy **giao của cả ba archive**. `geometry_fused_train.tar` đã được inspect trực tiếp với contract:

```text
geometry_fused/train/<raw_id>.npz
keys: R_G, C_G
shape mẫu: 374×1238, float32
C_G: [0,1]
```

Lưu ý tên trong archive có thể là `..._sync_image_03_0000000915`, trong khi dataset dùng `..._sync_image_0000000915_image_03`. Script extraction canonicalize thứ tự này cho metric/DA/geometry trước khi lấy giao ID; không được chỉ đổi tên thư mục rồi kỳ vọng loader tự tìm thấy.

KITTI anonymous test có đúng 1.000 ảnh nhưng không có ground truth công khai. Vì vậy notebook xuất 1.000 PNG để nộp benchmark; metric local được tính trên 200 validation có GT.

## 1. The Optimal Method: Zipped Files + `gdown`
Never read unzipped data directly from a mounted Google Drive (`/content/drive/`) during model training. The FUSE filesystem overhead will severely throttle training speed. 

**Standard Operating Procedure:**
1. Archive your dataset into a single `.zip` or `.tar` file before uploading to Google Drive.
2. Set the file sharing permission to **"Anyone with the link"**.
3. Extract the **File ID** from the share link (the string between `/d/` and `/view`).
4. Download directly to the local Colab disk using `gdown` and extract locally.

```bash
# Install gdown if necessary
!pip install gdown

# Download the file
# Metric coarse
!gdown 1ZtRVY67l3QkdgegSxDYtf6Cq_l-_BjQW

# Depth Anything V2
!gdown 1DOtv8E_zW-pOkms2XUqro9vVfMnkqzbB

# Extract locally
!tar -xf ./metric_coarse_train.tar -C /content/data
!tar -xf ./depth_anything_train_raw.tar -C /content/data

```

## 2. Bypassing Google Drive Download Limits

If you encounter the `Too many users have viewed or downloaded this file recently` error, Google has locked the file's API access due to bandwidth quotas.

**The Fix (Make a Copy):**

1. Open the locked public Drive link in your browser.
2. Click **"Add shortcut to Drive"** or **"Make a copy"**.
3. Go to your Google Drive, find the copied file, and set permissions to **"Anyone with the link"**.
4. Use the new File ID with `gdown`. This resets the quota because the file is now hosted on your account.

## 3. Handling Massive Unzipped Directories

If you absolutely cannot zip the data beforehand and must transfer thousands of individual files, bypass `gdown` and `drive.mount()`. Use `rclone`.

`rclone` executes highly concurrent transfers and bypasses Drive FUSE overhead. Configuration requires setting up a remote inside the notebook, but it is the fastest method for raw directory transfer.

## 4. Long-Term Recommendation: Migrate to Hugging Face

Google Drive is not a CDN. Continual dataset pulls will trigger rate limits.

For stable machine learning pipelines, host datasets on Hugging Face Hub (supports private repositories). It requires zero complex authentication in Colab and has no bandwidth throttling.

```bash
# Pull datasets instantly without rate limits
!wget [https://huggingface.co/datasets/username/repo/resolve/main/dataset.tar](https://huggingface.co/datasets/username/repo/resolve/main/dataset.tar)

```
