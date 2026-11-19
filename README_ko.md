# SuperMimic Gym

[English](README.md) | **한국어**

[SuperMimic](https://arxiv.org/abs/2610.28480)으로 Unitree G1의 전신 모션 트래킹을 학습합니다. 리플레이로 미분 가능한 월드 모델을 학습하고, 상상 롤아웃을 통해 트래킹 보상을 역전파하여 목표 조건부 정책을 학습합니다. 구조는 [legged-gym-mujoco](https://github.com/JihoonPark20/legged-gym-mujoco)를 따릅니다. 물리는 멀티스레드 CPU MuJoCo에서, 월드 모델·정책·크리틱은 CUDA, MPS 또는 CPU에서 학습하므로 같은 코드가 Linux와 macOS에서 동작합니다.

## 설치

```bash
python -m pip install -e .                      # Python 3.10–3.13; NVIDIA 환경은 CUDA용 PyTorch를 먼저 설치
python -m legged_gym.scripts.prepare_motions    # LAFAN1 40개 클립(SHA256 검증) -> data/motions/lafan1
```

## 학습

```bash
python -m legged_gym.scripts.train                                   # 환경 1024개, 21000스텝
python -m legged_gym.scripts.train --resume --max_iterations 10000   # 최신 체크포인트부터 이어서 학습
```

로그와 체크포인트(`model_*.pt`, 약 216 MB, 500스텝마다)는 `logs/g1_mimic/`에 저장됩니다. 1 iteration은 벡터화된 환경 1스텝과 학습기 업데이트 0.4회입니다. 학습 장치는 자동 선택(CUDA, MPS, CPU 순)되며 `--rl_device`로 지정할 수 있습니다. 설정은 [g1_mimic_config.py](legged_gym/envs/g1_mimic/g1_mimic_config.py)에 있습니다.

## 재생과 평가

```bash
mjpython -m legged_gym.scripts.play --checkpoint_path checkpoints/g1_mimic_lafan1.pt --motion jumps1_subject1   # macOS 뷰어
python -m legged_gym.scripts.play --checkpoint_path checkpoints/g1_mimic_lafan1.pt --evaluate                   # 40개 클립 벤치마크
python -m legged_gym.scripts.export_policy logs/g1_mimic/<run>/model_21000.pt checkpoints/my_policy.pt          # 정책만 저장
python -m pytest tests
```

Linux에서는 뷰어를 `python`으로 실행합니다. `--evaluate`는 모든 클립을 첫 프레임부터 노이즈·무작위화 없이 한 번씩 재생하고 결과 JSON을 체크포인트 옆에 저장합니다. `--checkpoint_path` 대신 `--resume`을 쓰면 최신 학습 체크포인트를 사용합니다.

## 물리는 CPU, 학습은 가속기

Apple M1 Max, 환경 1024개, 배치 512 실측입니다. 1 iteration은 제어 1스텝과 업데이트 0.4회입니다.

| 구성 | 물리 | 업데이트 | iteration당 |
|---|---:|---:|---:|
| **이 저장소: MuJoCo CPU 10스레드 + MPS** | 38 ms | 1.22 s | **0.53 s** |
| MuJoCo Warp CPU 디바이스 + MPS (macOS의 mjlab) | 1101 ms | 1.22 s | 1.59 s |
| MuJoCo CPU + CPU 학습기 | 38 ms | 7.94 s | 3.21 s |

CUDA가 없으면 MuJoCo Warp는 CPU 디바이스에서 실행되며 측정 시 단일 코어만 사용했습니다. 같은 모델과 솔버 설정에서 네이티브 `mujoco.batch` 10스레드가 29배 빨랐습니다. 학습기는 넓은 MLP와 32스텝 역전파가 대부분이라 MPS(`torch.compile`)가 CPU보다 6.5배 빨랐습니다. NVIDIA GPU에서는 MuJoCo Warp가 GPU에서 시뮬레이션할 수 있지만, 이 저장소는 모든 플랫폼에서 하나의 코드 경로를 쓰도록 물리를 CPU에 둡니다.

## 구성

```
legged_gym/envs/g1_mimic/   태스크 설정과 환경
legged_gym/supermimic/      월드 모델, 상상 학습, 정책·크리틱, 리플레이, 러너, 벤치마크
legged_gym/scripts/         train, play, prepare_motions, export_policy
legged_gym/sim/             MuJoCo 모델 구성과 배치 시뮬레이션
resources/                  G1 모델(Unitree), LAFAN1 매니페스트
```

## 라이선스

코드: [BSD-3](LICENSE). SuperMimic 구현은 [supermimic-m5](https://github.com/WXyuany/supermimic-m5)([MIT](licenses/supermimic_m5_LICENSE))에서 이식했고, 로봇 모델과 설정 출처는 [licenses](licenses)에 있습니다. LAFAN1은 CC BY-NC-ND 4.0으로, 재배포하지 않고 다운로드하며 이 데이터로 학습한 정책은 비상업 연구용입니다.
