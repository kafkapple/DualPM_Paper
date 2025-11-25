# DualPM 실행 가이드

**날짜:** 2025-11-23
**작성자:** Claude Code
**목적:** DualPM 프로젝트의 전체 워크플로우 단계별 실행 가이드

---

## 개요

DualPM(Dual Point Maps)은 단일 2D 이미지로부터 3D 형상을 재구성하는 딥러닝 모델입니다.

**핵심 아이디어:**
- **입력:** 단일 RGB 이미지
- **출력:** 3D point cloud (canonical + posed)
- **방법:** Dual Pointmap representation (여러 depth layer의 결합)

---

## 전체 워크플로우

```
원본 데이터
    ↓
[Step 1] 전처리: Mesh → Dual Pointmap (ground truth 생성)
    ↓
[Step 2] 학습: 2D Image → 3D Pointmap 예측 모델 학습
    ↓
[Step 3] 추론: 새로운 이미지로 3D 재구성
    ↓
[Step 4] 시각화: Point cloud 시각화
```

---

## 환경 설정

### CUDA 환경 변수 (필수)

DualPM은 nvdiffrast를 사용하므로 CUDA 경로를 명시적으로 설정해야 합니다.

#### Wrapper 스크립트 생성 (권장)

```bash
cd ~/dev/DualPM_Paper

cat > run_with_cuda.sh << 'EOF'
#!/bin/bash
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_PATH=/usr/local/cuda-11.8
export CUDACXX=/usr/local/cuda-11.8/bin/nvcc
export C_INCLUDE_PATH=/usr/local/cuda-11.8/include
export CPLUS_INCLUDE_PATH=/usr/local/cuda-11.8/include
export CPATH=/usr/local/cuda-11.8/include
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH

# 캐시 삭제 (첫 실행 시)
if [ "$1" == "--clean" ]; then
    rm -rf ~/.cache/torch_extensions/
    rm -rf /tmp/torch_extensions/
    shift
fi

python "$@"
EOF

chmod +x run_with_cuda.sh
```

**사용법:**
```bash
# 첫 실행 시 (캐시 삭제)
./run_with_cuda.sh --clean scripts/train.py

# 이후 실행
./run_with_cuda.sh scripts/train.py
```

---

## Step 1: 전처리 (Rasterization)

### 목적

원본 mesh 데이터를 dual pointmap 형태로 변환하여 학습 속도를 향상시킵니다.

### 실행

```bash
cd ~/dev/DualPM_Paper

# Conda 환경 활성화
conda activate dualpm

# 전처리 실행 (약 20분 소요)
./run_with_cuda.sh --clean scripts/raster_pointmaps.py configs/main.yaml
```

### 출력

**위치:** `/home/joon/data/dualpm_dataset/pointmaps_160/`

**내용:**
- 8334개의 `.npz` 파일
- 각 파일은 하나의 샘플에 대한 ground truth dual pointmap

**NPZ 파일 구조:**
```python
data = np.load('pointmaps_160/sample_0001.npz')

# 포함 데이터
values: np.ndarray   # shape: (N, 7)
                     # [x, y, z, nx, ny, nz, mask]
                     # N: non-zero 포인트 개수

indices: np.ndarray  # shape: (N, 3)
                     # [layer_idx, height, width]
                     # Sparse 인덱스

shape: tuple         # (num_layers, H, W, 7)
                     # 원본 dense tensor의 shape
```

### 진행 상황 확인

```bash
# 생성된 파일 개수 확인
ls /home/joon/data/dualpm_dataset/pointmaps_160/ | wc -l
# 출력: 8334

# 파일 크기 확인
du -sh /home/joon/data/dualpm_dataset/pointmaps_160/
```

---

## Step 2: 모델 학습 (Training)

### 목적

2D 이미지 feature로부터 3D dual pointmap을 예측하는 모델을 학습합니다.

### 실행

```bash
cd ~/dev/DualPM_Paper

# 학습 시작 (GPU 필수)
./run_with_cuda.sh scripts/train.py
```

### 학습 설정 (configs/main.yaml)

```yaml
train_config:
  steps: 100000           # 총 학습 step
  save_every: 5000        # 5000 step마다 checkpoint 저장
  val_every: 20           # 20 step마다 validation
  log_every: 10           # 10 step마다 loss 로그
  gradient_clip_value: 1.0
  batch_size: 12
  save_path: ./save_path  # Checkpoint 저장 경로
```

### 출력

**위치:** `./save_path/`

**파일:**
```
save_path/
├── weights_5000.pth
├── weights_10000.pth
├── weights_15000.pth
├── ...
└── weights_100000.pth
```

**Checkpoint 내용:**
```python
checkpoint = torch.load('save_path/weights_50000.pth')

# 포함 데이터
model_state: dict      # 모델 가중치
optim_state: dict      # Optimizer 상태
scheduler_state: dict  # LR Scheduler 상태
```

### 학습 모니터링

**터미널 출력 예시:**
```
Training loss: 0.0234, Val loss: 0.0456: 12%|████      | 12000/100000 [2:34:12<18:23:45,  7.05it/s]
```

**실시간 로그 확인:**
```bash
# 다른 터미널에서
watch -n 5 'ls -lh save_path/'
```

**예상 소요 시간:**
- GPU: RTX 3090 기준 약 10-15시간 (100K steps)
- Batch size, 데이터 크기에 따라 변동

### Background 실행 (권장)

장시간 학습 시 background로 실행:

```bash
# nohup으로 background 실행
nohup ./run_with_cuda.sh scripts/train.py > training.log 2>&1 &

# 프로세스 ID 확인
echo $!

# 로그 실시간 확인
tail -f training.log

# 학습 중단 (필요 시)
kill [PID]
```

### Resume 학습

중단된 학습을 이어서 진행:

```yaml
# configs/main.yaml 수정
train_config:
  use_weights: ./save_path/weights_50000.pth  # 이어갈 checkpoint
```

```bash
./run_with_cuda.sh scripts/train.py
```

---

## Step 3: 추론 (Inference)

### 목적

학습된 모델로 새로운 이미지에서 3D point cloud를 재구성합니다.

### Config 파일 (configs/infer.yaml)

```yaml
# configs/infer.yaml 예시
weights_path: ./save_path/weights_100000.pth
device: cuda
confidence_threshold: 0.5
output_dir: ./inference_results

dataset:
  _target_: dualpm_paper.alt_dataset.RasterizeDataset
  root: /home/joon/data/dualpm_dataset
  image_size: 160
  num_layers: 4
  input_mode: sd_dino

dataloader:
  _target_: torch.utils.data.DataLoader
  batch_size: 4
  num_workers: 4
  shuffle: false
```

### 실행

```bash
cd ~/dev/DualPM_Paper

# 추론 실행
./run_with_cuda.sh scripts/infer.py --config-name infer
```

### 출력

**위치:** `./inference_results/`

**파일:**
```
inference_results/
├── sample_0001_canon.ply   # Canonical pose (정규화된 형태)
├── sample_0001_rec.ply     # Reconstructed pose (재구성된 형태)
├── sample_0002_canon.ply
├── sample_0002_rec.ply
└── ...
```

**PLY 파일 구조:**
- ASCII format point cloud
- Vertex positions (x, y, z)
- 일반적인 3D 뷰어로 열기 가능

---

## Step 4: 시각화

### 방법 1: MeshLab (GUI)

```bash
# MeshLab 설치 (Ubuntu)
sudo apt install meshlab

# 실행
meshlab inference_results/sample_0001_rec.ply
```

### 방법 2: CloudCompare (GUI)

```bash
# CloudCompare 설치
sudo snap install cloudcompare

# 실행
cloudcompare.CloudCompare inference_results/sample_0001_rec.ply
```

### 방법 3: Python 스크립트

```python
# visualize_pointcloud.py
import trimesh
import numpy as np

# PLY 파일 로드
pc = trimesh.load('inference_results/sample_0001_rec.ply')

# 점 개수 확인
print(f"Point count: {len(pc.vertices)}")

# 3D 시각화 (matplotlib)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

vertices = pc.vertices
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
           c=vertices[:, 2], s=1, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Reconstructed Point Cloud')
plt.tight_layout()
plt.savefig('visualization.png', dpi=150)
plt.show()
```

실행:
```bash
python visualize_pointcloud.py
```

### 방법 4: NPZ 파일 직접 시각화

전처리 단계에서 생성된 npz 파일 시각화:

```python
# visualize_npz.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# NPZ 로드
data = np.load('/home/joon/data/dualpm_dataset/pointmaps_160/sample_0001.npz')
values = data['values']  # (N, 7): [x, y, z, nx, ny, nz, mask]
indices = data['indices']  # (N, 3): [layer, h, w]

print(f"Point count: {len(values)}")
print(f"Shape: {data['shape']}")

# 3D 좌표 추출
xyz = values[:, :3]
normals = values[:, 3:6]

# 시각화
fig = plt.figure(figsize=(15, 5))

# Point cloud
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
            c=xyz[:, 2], s=1, cmap='viridis')
ax1.set_title('Point Cloud')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Layer distribution
ax2 = fig.add_subplot(132)
layer_counts = np.bincount(indices[:, 0])
ax2.bar(range(len(layer_counts)), layer_counts)
ax2.set_title('Points per Layer')
ax2.set_xlabel('Layer Index')
ax2.set_ylabel('Point Count')

# Normal vectors (sample)
ax3 = fig.add_subplot(133, projection='3d')
sample_idx = np.random.choice(len(xyz), 1000, replace=False)
xyz_sample = xyz[sample_idx]
normals_sample = normals[sample_idx]
ax3.quiver(xyz_sample[:, 0], xyz_sample[:, 1], xyz_sample[:, 2],
           normals_sample[:, 0], normals_sample[:, 1], normals_sample[:, 2],
           length=0.05, arrow_length_ratio=0.3, alpha=0.5)
ax3.set_title('Surface Normals (sampled)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.tight_layout()
plt.savefig('npz_visualization.png', dpi=150)
plt.show()
```

---

## 문제 해결 (Troubleshooting)

### 1. CUDA 헤더 파일 찾을 수 없음

**오류:**
```
fatal error: cuda_runtime_api.h: No such file or directory
```

**해결:**
```bash
# 환경 변수 확인
echo $CUDA_HOME
echo $C_INCLUDE_PATH

# wrapper 스크립트 사용 확인
./run_with_cuda.sh --clean [script]

# 또는 직접 환경 변수 설정
export CUDA_HOME=/usr/local/cuda-11.8
export C_INCLUDE_PATH=/usr/local/cuda-11.8/include
export CPLUS_INCLUDE_PATH=/usr/local/cuda-11.8/include
```

### 2. GCC 버전 오류

**오류:**
```
error: "You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later."
```

**해결:**
```bash
# Conda 환경에 GCC 설치
conda activate dualpm
conda install -c conda-forge gxx_linux-64=11.2.0

# 확인
gcc --version  # 11.x.x
```

### 3. OOM (Out of Memory)

**오류:**
```
RuntimeError: CUDA out of memory
```

**해결:**
```yaml
# configs/main.yaml 수정
train_config:
  batch_size: 6  # 12 → 6으로 감소

resolution: 128  # 160 → 128로 감소
```

### 4. Dataset not found

**오류:**
```
FileNotFoundError: Dataset path not found
```

**해결:**
```bash
# dataset_root 경로 확인
ls /home/joon/data/dualpm_dataset/

# configs/main.yaml 수정
dataset_root: '/home/joon/data/dualpm_dataset'  # 절대 경로 사용
```

### 5. nvdiffrast 컴파일 실패

**해결:**
```bash
# 캐시 완전 삭제 후 재설치
rm -rf ~/.cache/torch_extensions/
rm -rf /tmp/torch_extensions/
pip uninstall nvdiffrast -y
pip install git+https://github.com/NVlabs/nvdiffrast/

# 환경 변수와 함께 Python import 테스트
./run_with_cuda.sh -c "import nvdiffrast.torch as dr; print('OK')"
```

---

## 디렉토리 구조

```
DualPM_Paper/
├── configs/
│   ├── main.yaml           # 학습/전처리 설정
│   └── infer.yaml          # 추론 설정
├── scripts/
│   ├── raster_pointmaps.py # Step 1: 전처리
│   ├── train.py            # Step 2: 학습
│   ├── infer.py            # Step 3: 추론
│   └── estimate_skeleton.py
├── src/
│   └── dualpm_paper/       # 모델 코드
├── docs/
│   └── reports/
│       └── 251123_dualpm_workflow_guide.md  # 이 문서
├── save_path/              # 학습된 weights (생성됨)
│   ├── weights_5000.pth
│   └── ...
├── inference_results/      # 추론 결과 (생성됨)
│   ├── sample_0001_canon.ply
│   └── sample_0001_rec.ply
├── run_with_cuda.sh        # CUDA wrapper 스크립트
└── README.md
```

**외부 데이터:**
```
/home/joon/data/dualpm_dataset/
├── images/                 # 입력 이미지
├── poses/                  # Pose 데이터
├── shapes/                 # 원본 mesh
├── metadata/               # 메타데이터
└── pointmaps_160/          # 생성된 dual pointmaps (Step 1 출력)
    ├── sample_0001.npz
    ├── sample_0002.npz
    └── ... (8334 files)
```

---

## 실행 체크리스트

### 전처리 (Step 1)
- [ ] Conda 환경 활성화: `conda activate dualpm`
- [ ] `configs/main.yaml`에서 `dataset_root` 확인
- [ ] `run_with_cuda.sh` 생성 및 권한 설정
- [ ] `./run_with_cuda.sh --clean scripts/raster_pointmaps.py configs/main.yaml` 실행
- [ ] `/home/joon/data/dualpm_dataset/pointmaps_160/` 생성 확인
- [ ] 8334개 npz 파일 생성 확인

### 학습 (Step 2)
- [ ] `configs/main.yaml`에서 `save_path` 설정 확인
- [ ] GPU 사용 가능 확인: `nvidia-smi`
- [ ] Background 실행: `nohup ./run_with_cuda.sh scripts/train.py > training.log 2>&1 &`
- [ ] 로그 확인: `tail -f training.log`
- [ ] Checkpoint 저장 확인: `ls -lh save_path/`
- [ ] Loss 감소 추이 모니터링

### 추론 (Step 3)
- [ ] `configs/infer.yaml` 생성 및 설정
- [ ] `weights_path` 경로 확인
- [ ] `./run_with_cuda.sh scripts/infer.py --config-name infer` 실행
- [ ] `inference_results/` 디렉토리 생성 확인
- [ ] `.ply` 파일 생성 확인

### 시각화 (Step 4)
- [ ] MeshLab 또는 CloudCompare 설치
- [ ] PLY 파일 열기 및 3D 형상 확인
- [ ] Canonical vs Reconstructed 비교
- [ ] 재구성 품질 평가

---

## 참고 자료

### DualPM 논문
- **제목:** Dual Point Maps for Single-Image 3D Reconstruction
- **링크:** [논문 링크 추가 필요]

### 관련 도구
- **nvdiffrast:** https://github.com/NVlabs/nvdiffrast
- **PyTorch:** https://pytorch.org/
- **Hydra:** https://hydra.cc/

### 시각화 도구
- **MeshLab:** https://www.meshlab.net/
- **CloudCompare:** https://www.cloudcompare.org/
- **trimesh:** https://trimsh.org/

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|-----------|
| 2025-11-23 | 1.0 | 초안 작성 (전처리 완료, 학습 준비) |

---

## 연락처

**문의:** Claude Code를 통한 지원
**프로젝트:** DualPM_Paper
**환경:** Ubuntu 18.04, CUDA 11.8, PyTorch 2.0.0
