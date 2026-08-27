<h1 align="center">Ling-3.0-Flash SGLang for DGX Spark</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Self-hosted OpenAI-compatible endpoint for Ling-3.0-flash on **one NVIDIA DGX Spark** (GB10, SM121, TP1, ~128 GB unified memory), served with [SGLang](https://github.com/sgl-project/sglang) in Docker and accelerated with the [inclusionAI/Ling-3.0-flash-dspark](https://huggingface.co/inclusionAI/Ling-3.0-flash-dspark) speculative draft.

Default path is the official GB10 **MXFP4 + DSPARK** cell from [sgl-project/sglang#36364](https://github.com/sgl-project/sglang/pull/36364) (`inclusionAI/Ling-3.0-flash-fp4` + `--moe-runner-backend flashinfer_mxfp4`). INT4 remains a one-flag fallback. `start.sh` and `stop.sh` are the only moving parts: they pull a prebuilt runtime, cache weights, and serve.

> **Note on Ling-3.0-flash-dspark:** it is **not a standalone model** — it's a 1.36B-parameter, 5-layer **DSpark speculative-decoding draft** (trained with [SpecForge](https://github.com/sgl-project/SpecForge), extending [DFlash](https://github.com/z-lab/dflash) with target-model auxiliary features and a confidence head). It is served *alongside* the MXFP4 or INT4 target via `--speculative-algorithm DSPARK --speculative-draft-model-path …` and typically accepts **~5.3 tokens per verification step** (macro mean across nine benchmarks; e.g. 6.4 on GSM8K, 3.5 on Alpaca).

## Quick start

```bash
# 1. Download MXFP4 weights + DSpark draft (pinned revisions)
./start.sh --download-only

# 2. Start the server (pulls LMSYS Ling image; defaults: QUANT=mxfp4,
#    DSPARK, 256k context, 6 concurrent, mem-fraction 0.85)
./start.sh

# Alternative profiles:
QUANT=int4 ./start.sh          # previous proven INT4 path
SPEC_ALGO=nextn ./start.sh     # built-in MTP (no extra checkpoint)
SPEC_ALGO=off ./start.sh       # no spec decode (high-throughput)

# 3. Verify
curl -fsS http://127.0.0.1:8888/v1/models

# 4. Chat (OpenAI-compatible)
curl -sS -m 180 http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hi in one short sentence."}],"max_tokens":64,"temperature":0.6,"top_p":0.95,"top_k":20}'

# 5. Stop
./stop.sh
```

### Prebuilt runtime (required for new users — no private git)

InclusionAI’s INT4 card still points at `github.com/inclusionAI/sglang_ling_v3`, which is often **private / 404**. **Do not rely on that clone.** `start.sh` pulls a **public** prebuilt image:

| Image | When |
|---|---|
| **`lmsysorg/sglang:dev-Ling-3.0-flash`** (default for `QUANT=mxfp4` and `SPEC_ALGO=dspark`) | Official LMSYS Ling runtime — DSPARK + FlashInfer CUTLASS MXFP4, multi-arch including arm64 |
| **`ghcr.io/miaai-lab/ling-3.0-flash-sglang-dgx-spark:ling_v3_support`** (`QUANT=int4` with `SPEC_ALGO=nextn`/`off`) | Public GHCR — NGC PyTorch + baked `ling_v3_support` (INT4 on Spark; predates DSPARK and MXFP4) |

```bash
# Default (MXFP4 + DSPARK → LMSYS image, public — no docker login)
./start.sh

# Proven INT4 fallback
QUANT=int4 ./start.sh

# Force the Spark GHCR image (INT4 + NEXTN/off only)
QUANT=int4 SPEC_ALGO=nextn IMAGE=ghcr.io/miaai-lab/ling-3.0-flash-sglang-dgx-spark:ling_v3_support ./start.sh
```

Package: https://github.com/users/MiaAI-Lab/packages/container/package/ling-3.0-flash-sglang-dgx-spark

## Requirements

- Linux with **Docker** (`docker` in `PATH`) and `curl`
- An NVIDIA GPU with enough unified/VRAM memory (~120 GB+ recommended for this model class)
- CUDA-capable Docker runtime (`--gpus all`)
- Free disk for model weights + a prebuilt runtime image (~80–110 GB: MXFP4 or INT4 checkpoint + 2.7 GB DSpark draft + ~30 GB image)

## Preflight checks

`start.sh` fails fast with a clear message (not a cryptic crash) if the new machine isn't ready:

- Docker daemon is reachable
- NVIDIA driver present (`nvidia-smi -L`)
- NVIDIA container runtime installed (required for `--gpus all`) — hint for `nvidia-container-toolkit` if missing
- Host RAM is at least 80 GiB (warns below 110 GiB, the DGX Spark class)
- 40 GiB+ free disk in the script directory (warns below 100 GiB)
- Ports are free: the server port and internal dist port 2345

If a server is already running on the same port, it exits with a hint to run `./stop.sh` first.

## Environment variables

All optional. Defaults are conservative and tuned for a single Spark host.

| Variable | Default | Description |
|---|---|---|
| `QUANT` | `mxfp4` | `mxfp4` (`inclusionAI/Ling-3.0-flash-fp4` + FlashInfer MXFP4) or `int4` (`inclusionAI/Ling-3.0-flash-int4`) |
| `SPEC_ALGO` | `dspark` | Speculative decoding: `dspark` (DSPARK + external Ling-3.0-flash-dspark draft), `nextn` (built-in MTP), or `off` |
| `REPLAYSSM_CACHE_LEN` | `32` | `--linear-replayssm-cache-len` for DSPARK (power of two ≥ 2× verify window 9) |
| `ENABLE_NEXTN` | *(legacy)* | Backward-compat override: `0` → `SPEC_ALGO=off`, `1` → `nextn` (ignored when `SPEC_ALGO` is set) |
| `PORT` | `8888` | Server port |
| `CTX` | `262144` (256k) | Context length |
| `MEM_FRACTION_STATIC` | `0.85` (mxfp4) / `0.75` (int4) | Fraction of the unified-memory pool for weights + KV. Official GB10 cell uses 0.85. Do not push to 0.87 unless nothing else is running on the box. |
| `MAX_RUNNING_REQUESTS` | `6` | Max concurrent requests |
| `MAX_MAMBA_CACHE_SIZE` | `32` | Mamba cache size (16 is a safer first run) |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | KV cache dtype; set to empty string to omit the flag |
| `DOCKER_MEMORY` | *(unset)* | Cap the container, e.g. `100g` — preferred on unified-memory hosts so the container dies before the host OOMs |
| `IMAGE` | `lmsysorg/sglang:dev-Ling-3.0-flash` | Prebuilt SGLang (public; no private git). GHCR `ling_v3_support` is INT4 + NEXTN/off only. |
| `USE_LMSYS_IMAGE` | `0` | Set `1` to force `lmsysorg/sglang:dev-Ling-3.0-flash` |
| `MODEL_REVISION` | pinned per `QUANT` | Hugging Face snapshot for the target |
| `DSPARK_REVISION` | `8e5d9988…` | Hugging Face snapshot for the DSpark draft |
| `HF_TOKEN` | *(empty)* | Hugging Face token for gated models |
| `HF_HOME` | `~/.cache/huggingface` | Where weights are cached |
| `FORCE_SOURCE_BUILD` | `0` | Set `1` only to rebuild from `SGLANG_REPO` (usually unnecessary) |

**Default launch (official GB10 MXFP4 cell + Spark extras):** `QUANT=mxfp4 SPEC_ALGO=dspark CTX=262144 MAX_RUNNING_REQUESTS=6 MEM_FRACTION_STATIC=0.85`, plus `--tool-call-parser ling3 --reasoning-parser ling3 --chunked-prefill-size 8192`. For resource-constrained hosts, pull back with `CTX=8192 MAX_RUNNING_REQUESTS=1 SPEC_ALGO=off`.

```bash
# Matches the official GB10 low-latency DSPARK cell, with parsers enabled
./start.sh
```

KV pool vs `MEM_FRACTION_STATIC` (measured on this host, fp8_e4m3, DSPARK):

| `MEM_FRACTION_STATIC` | KV pool (tokens) | KV memory | Free GPU-side mem after graphs | Host `MemAvailable` |
|---|---|---|---|---|
| 0.75 | 237,922 | ~9.1 GB | 26.8 GB | ~24 GB |
| 0.80 | 328,898 | ~12.5 GB | 22.0 GB | ~20 GB |
| 0.87 | 576,930 | ~23.2 GB | 10.2 GB | ~9.5 GB |

## Endpoint

Once the container reports ready, a single server listens on `0.0.0.0:8888/v1` (host network):

- `GET /v1/models`
- `POST /v1/chat/completions`

Default request config (recommended — matches the server's defaults):

```json
{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": true}}
```

**Thinking mode:** Ling's chat template defaults thinking **off**. The `ling3` parser is gated on that same flag, so with it off, thinking text and a stray `</think>` land in `content`. `start.sh` sets `--default-chat-template-kwargs '{"enable_thinking":true,"thinking":true}'` so reasoning is split server-side. Disable per request for fast non-reasoning answers:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

Sampling is **not** defaulted by the server. Clients should send `temperature=0.6, top_p=0.95, top_k=20` (model-card defaults). Unset falls back to OpenAI `temperature=1.0`.

## Performance

### Official GB10 MXFP4 (sgl-project/sglang#36364, not yet merged)

Measured on a real GB10 with `inclusionAI/Ling-3.0-flash-fp4`, `--moe-runner-backend flashinfer_mxfp4`, `lmsysorg/sglang:dev-Ling-3.0-flash`. Speed bench is ISL 8192 / OSL 1024 (tok/s **includes prefill**). DSPARK decode TPOT at c=1 is 9.48 ms (~105 tok/s). Full GSM8K (1319): DSPARK 96.36%, spec-off 96.82%.

| Cell | Conc. | TTFT | TPOT | tok/s (prefill+decode) |
|---|---|---|---|---|
| DSPARK | 1 | 2.17 s | 9.48 ms | 580 |
| DSPARK | 16 | 71.3 s | 38.1 ms | 1157 |
| spec off | 1 | 2.07 s | 25.8 ms | 324 |
| spec off | 16 | 21.7 s | 106.5 ms | 1129 |

This repo has not re-run that MXFP4 table locally yet. `QUANT=int4` remains the numbers we measured on this host.

### Previous (INT4 DSPARK, `lmsysorg/sglang:dev-Ling-3.0-flash`, `MEM_FRACTION_STATIC=0.87`)

With the external Ling-3.0-flash-dspark draft active, decode throughput on this DGX Spark (GB10), `--chunked-prefill-size 8192`, fp8_e4m3 KV:

| Concurrency | Aggregate (tok/s) | Per-request (tok/s) | TTFT |
|---|---|---|---|
| ×1 | 83 | 83 | 1.44 s |
| ×2 | 84 | 55 | 482 ms |
| ×4 | 107 | 33 | 1.75 s |
| ×6 | 136 | 29 | 529 ms |

vs the NEXTN baseline below: **2.2× faster at ×1** (83 vs 37 tok/s) and **1.8× at ×6** (136 vs 76 tok/s). Observed **accept length 5.2–6.4** (accept rate 0.53–0.67), consistent with the draft card's reported macro mean of 5.29 across nine benchmarks (6.40 GSM8K, 6.57 HumanEval, 3.51 Alpaca). Startup: ~160 s weight load + ~2 min draft/verify CUDA-graph capture.

### Baseline (NEXTN MTP, INT4 `ling_v3_support`, `MEM_FRACTION_STATIC=0.75`)

Decode throughput for `inclusionAI/Ling-3.0-flash-int4` on this DGX Spark (GB10), measured during the bring-up baseline (`MEM_FRACTION_STATIC=0.75`, `--chunked-prefill-size 8192`):

| Concurrency | Aggregate (tok/s) | Per-request (tok/s) | TTFT |
|---|---|---|---|
| ×1 | 37 | 37 | 220 ms |
| ×2 | 54 | 28 | 383 ms |
| ×4 | 60 | 28 | 421 ms |
| ×5 | 65 | 27 | 350 ms |
| ×6 | 76 | 26 | 4.99 s |

**agg** = server-wide decoded tokens per second; **str** = per-request tokens per second; **TTFT** = time to first token.

Note the ×6 spike: 4.99 s TTFT at 6 concurrent requests — batch efficiency degrades past ×5 under this configuration. TTFT under DSPARK is more erratic (1.44 s at ×1 vs 529 ms at ×6) but throughput scales much better; aggregate decode grows monotonically with concurrency instead of flattening.

## Files

| File | Purpose |
|---|---|
| `start.sh` | Download model, pull public runtime image, launch container, wait for readiness |
| `stop.sh` | Confirm and stop/remove the container; cleans up PID + tmp files |

## How it works

1. `start.sh` caches the MXFP4 (`Ling-3.0-flash-fp4`) or INT4 weights (and, for DSPARK, the 2.7 GB `Ling-3.0-flash-dspark` draft) under `$HF_HOME` at pinned revisions (uses `hf`, `huggingface-cli`, or falls back to a Docker download).
2. It pulls a **public prebuilt** image with SGLang already installed (no `git clone` of the private InclusionAI fork). MXFP4 and DSPARK use `lmsysorg/sglang:dev-Ling-3.0-flash` because FlashInfer MXFP4 and `dspark_components` postdate the `ling_v3_support` build.
3. The container runs with `--network host`, `--ipc host`, `--gpus all`, `--shm-size=32g`, and the model snapshot from the HF cache.
4. `start.sh` tails the logs and blocks until `/v1/models` responds, then prints the endpoint.

`.sglang.pid` holds the container ID, `.sglang.log` the launch line; both live in this directory and are removed by `stop.sh`.

## Notes & safety

- **Unified memory**: model weights, compile and KV cache all share host RAM. MXFP4 defaults to **0.85** (official GB10 cell). INT4 defaults to **0.75**. An older INT4 profile at 0.87 left only ~9.5 GiB `MemAvailable` — do not use that unless the box is dedicated. If `MemAvailable` keeps falling below ~10 GB during load, stop with `./stop.sh` and relaunch at `MEM_FRACTION_STATIC≤0.80`. Consider `DOCKER_MEMORY=100g` so the container dies before the host OOMs.
- **Context & concurrency**: the script defaults are **256k context** and **6 concurrent requests** (`CTX=262144 MAX_RUNNING_REQUESTS=6`) — matching this host. Spec decode defaults to **DSPARK** (external draft, `SPEC_ALGO=dspark`); switch with `SPEC_ALGO=nextn` (built-in MTP) or `SPEC_ALGO=off`. At high batch saturation, spec-decode overhead can outweigh the speedup — prefer `SPEC_ALGO=off` for batch jobs. Constrained hosts can pull back per launch: `CTX=8192 MAX_RUNNING_REQUESTS=1 SPEC_ALGO=off`.

## Monitoring memory & GPU

This is a **unified-memory** machine (GB10): model weights, SGLang compile caches and the KV cache all draw from the same ~128 GB pool — there is **no discrete VRAM to query**. `nvidia-smi` therefore reports GPU compute utilization fine but returns `memory.used = [N/A]`.

```bash
# GPU utilization (memory columns are [N/A] on GB10 — expected)
nvidia-smi

# The number that actually matters on this host:
free -h            # watch MemAvailable, not the "free" column
watch -n2 free -h  # live view while the server is loading/serving
```

While loading, expect GPU utilization to rise and MemAvailable to fall as weights + KV cache fill host RAM. If MemAvailable drops below ~10 GB and keeps falling (or you see container restarts), stop with `./stop.sh` and relaunch with a leaner profile (`CTX=8192 MAX_RUNNING_REQUESTS=1 SPEC_ALGO=off`, `MEM_FRACTION_STATIC≤0.80`).

- `curl http://127.0.0.1:8888/v1/models` fails → `docker logs ling-3.0-flash` (first build takes minutes; startup-looking output will stream).
- Container vanished before ready → `./stop.sh` then relaunch with `DOCKER_MEMORY=100g` and see logs.
- Model download stalls → run `./start.sh --download-only --token <HF_TOKEN>` if it is a gated repo.

## License & credits

- Model weights: [inclusionAI/Ling-3.0-flash-fp4](https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4), [inclusionAI/Ling-3.0-flash-int4](https://huggingface.co/inclusionAI/Ling-3.0-flash-int4), and draft [inclusionAI/Ling-3.0-flash-dspark](https://huggingface.co/inclusionAI/Ling-3.0-flash-dspark) (InclusionAI) — check the model cards for their licenses
- SGLang: [sgl-project/sglang](https://github.com/sgl-project/sglang)
- DSpark extends [DFlash](https://github.com/z-lab/dflash); trained with [SpecForge](https://github.com/sgl-project/SpecForge)
- GB10 MXFP4 flags and numbers from [sgl-project/sglang#36364](https://github.com/sgl-project/sglang/pull/36364) (open as of 2026-08-27). Ling V3 + DSPARK landed in SGLang main via [#33561](https://github.com/sgl-project/sglang/pull/33561). The LMSYS `dev-Ling-3.0-flash` tag last published 2026-08-21; if MXFP4 output looks wrong on SM12x, rebuild/pull after the SwigluStep clamp from #33561, or fall back with `QUANT=int4`.