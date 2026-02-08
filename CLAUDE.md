# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Forcing is a training-free framework for long-form video generation using autoregressive video diffusion models. It combines **Deep Sink** (enlarged attention sink with temporal RoPE adjustment) and **Participative Compression** (attention-based token pruning) to achieve 12x length extrapolation (5s → 60s+) without fine-tuning. Built on top of the Wan2.1-T2V-1.3B model.

## Setup

```bash
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

Download models:
```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

## Running Inference

Deep Sink + Participative Compression (full quality):
```bash
bash DS_PC_inference.sh
```

Deep Sink only (faster):
```bash
bash DS_inference.sh
```

Direct invocation:
```bash
python inference.py \
    --config_path configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml \
    --output_folder ./output \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path ./prompts/MovieGenVideoBench_txt/line_0010.txt \
    --use_ema --is_ds_only 0
```

## Training

```bash
python train.py --config_path configs/self_forcing_dmd.yaml --logdir ./logs
```

Trainer variants selected by `config.trainer`: `DiffusionTrainer`, `GANTrainer`, `ODETrainer`, `ScoreDistillationTrainer`.

## Architecture

### Entry Points
- `inference.py` — Main inference script. Sets up distributed environment, loads model/config, runs pipeline.
- `train.py` — Main training script. Supports FSDP multi-GPU, mixed precision (bf16), W&B logging, EMA.

### Core Innovation (in `wan/modules/`)
- `causal_model.py` (~1500 lines) — **CausalWanModel**: Full Deep Sink + Participative Compression implementation. The heart of the project.
- `causal_model_DS.py` (~1100 lines) — **CausalWanModelDS**: Deep Sink only variant (simpler/faster).
- `model.py` — Base WanModel (non-causal, used as frozen scorer during training).

**Deep Sink**: Maintains ~50% of KV cache as persistent sink tokens. `_rope_time_delta_mul_()` adjusts temporal RoPE to keep sink tokens aligned with current generation timestep.

**Participative Compression (PC)**: `PCConfig` class configures top-C token selection based on fused attention scores from recent frames. Protected sinks are always retained; max-reuse limits prevent token starvation.

### Model Wrappers (`utils/wan_wrapper.py`)
- `WanDiffusionWrapper` — Wraps the diffusion backbone. In training, manages generator + real/fake scorers.
- `WanTextEncoder` — Wraps UM-T5-XXL text encoder.
- `WanVAEWrapper` — Video VAE encoder/decoder with pre-computed normalization.

### Pipelines (`pipeline/`)
- `causal_inference.py` — Few-step causal inference with KV-cache (uses `denoising_step_list`, e.g., 4 steps).
- `causal_diffusion_inference.py` — Multi-step diffusion inference (50 steps).
- `self_forcing_training.py` — Training pipeline with score distillation loss.
- `bidirectional_*.py` — Non-causal variants.

### Model Definitions (`model/`)
- `base.py` — `BaseModel` and `SelfForcingModel` base classes.
- `dmd.py` — Distribution Matching Distillation (primary training approach).
- Other variants: `causvid.py`, `gan.py`, `sid.py`, `ode_regression.py`.

### Configs (`configs/`)
- `default_config.yaml` — Base defaults.
- `self_forcing_dmd.yaml` — Main training config.
- `self_forcing_dmd/sink0-sink21.yaml` — Inference configs with varying sink sizes (10-14 recommended).

### Key Config Parameters
- `sink_size` (0-21): Attention sink token count
- `is_ds_only`: 1 = Deep Sink only, 0 = Deep Sink + PC
- `budget`: PC capacity multiplier (default 16)
- `recent`: PC recent window multiplier (default 4)
- `guidance_scale`: Classifier-free guidance strength (default 3.0)
- `denoising_step_list`: Timesteps for few-step inference (e.g., [1000, 750, 500, 250])

## Data Flow

```
Text Prompt → [T5 Encoder] → embeddings
                                ↓
Random Noise → [Causal Diffusion Generator] → Latents → [VAE Decoder] → MP4
                  ├─ Deep Sink (persistent KV cache)
                  ├─ PC (attention-based token pruning)
                  └─ Local windowed attention
```

Latent space: 1/8 spatial compression. Output: 480×832 pixels, 16 fps.

## Hardware Requirements
- Nvidia GPU with ≥24GB VRAM (RTX 3090, A6000, H100)
- 64GB RAM
- `demo_utils/memory.py` provides CPU-GPU swapping for constrained environments
