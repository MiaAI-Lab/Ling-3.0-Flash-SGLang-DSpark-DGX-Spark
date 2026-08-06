# Ling-3.0-Flash SGLang for DGX Spark

Self-hosted OpenAI-compatible endpoint for [inclusionAI/Ling-3.0-flash-int4](https://huggingface.co/inclusionAI/Ling-3.0-flash-int4) served with [SGLang](https://github.com/sgl-project/sglang) in Docker.

Designed and tested on a DGX Spark (GB10, SM121, ~128 GB unified memory) — the official INT4 recipe, no MXFP4/Cutlass fallbacks. `start.sh` and `stop.sh` are the only moving parts: they build, cache, and run everything.

## Quick start

```bash
# 1. Download the INT4 weights (also validates the image/tooling)
./start.sh --download-only

# 2. Start the server (conservative, Spark-safe defaults)
./start.sh

# 3. Verify
curl -fsS http://127.0.0.1:8888/v1/models

# 4. Chat (OpenAI-compatible)
curl -sS -m 180 http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hi in one short sentence."}],"max_tokens":64,"temperature":0.6,"chat_template_kwargs":{"enable_thinking":false}}'

# 5. Stop
./stop.sh
```

The first start compiles and installs SGLang (branch `ling_v3_support`) into `.sglang-persist/` — this takes several minutes. Subsequent starts reuse it.

## Requirements

- Linux with **Docker** (`docker` in `PATH`) and `curl`
- An NVIDIA GPU with enough unified/VRAM memory (~120 GB+ recommended for this model class)
- CUDA-capable Docker runtime (`--gpus all`)
- ~Free disk space for the model weights + a ~20 GB NGC PyTorch image (auto-pulled: `nvcr.io/nvidia/pytorch:26.01-py3`)

# Ping-pong: start.sh now checks the environment before doing anything
# and prints FATAL/WARN with actionable hints instead of cryptic errors.

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
| `PORT` | `8888` | Server port |
| `CTX` | `8192` | Context length. Raise only after a baseline run works |
| `MEM_FRACTION_STATIC` | `0.75` | SGLang static memory fraction (unified memory: stay conservative) |
| `MAX_RUNNING_REQUESTS` | `1` | Max concurrent requests |
| `MAX_MAMBA_CACHE_SIZE` | `32` | Mamba cache size (16 is a safer first run) |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | KV cache dtype; set to empty string to omit the flag |
| `ENABLE_NEXTN` | `0` | Set `1` to enable `--speculative-algorithm NEXTN` |
| `DOCKER_MEMORY` | *(unset)* | Cap the container, e.g. `100g` — preferred on unified-memory hosts so the container dies before the host OOMs |
| `IMAGE` | `nvcr.io/nvidia/pytorch:26.01-py3` | Docker image |
| `HF_TOKEN` | *(empty)* | Hugging Face token for gated models |
| `HF_HOME` | `~/.cache/huggingface` | Where weights are cached |

Example with a host memory cap and tighter settings:

```bash
MEM_FRACTION_STATIC=0.70 MAX_RUNNING_REQUESTS=1 MAX_MAMBA_CACHE_SIZE=16 CTX=8192 DOCKER_MEMORY=100g ./start.sh
```

## Endpoint

Once the container reports ready, a single server listens on `0.0.0.0:8888/v1` (host network):

- `GET /v1/models`
- `POST /v1/chat/completions`

Recommended client parameters (`chat_template_kwargs`):

```json
{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": true}}
```

Set `"enable_thinking": false` for faster/non-reasoning responses.

## Performance

Decode throughput for `inclusionAI/Ling-3.0-flash-int4` on this DGX Spark (GB10), measured with the container defaults (`MEM_FRACTION_STATIC=0.75`, `--chunked-prefill-size 8192`, INT4 `ling_v3_support` branch):

| Concurrency | Aggregate (tok/s) | Per-request (tok/s) | TTFT |
|---|---|---|---|
| ×1 | 37 | 37 | 220 ms |
| ×2 | 54 | 28 | 383 ms |
| ×4 | 60 | 28 | 421 ms |
| ×5 | 65 | 27 | 350 ms |
| ×6 | 76 | 26 | 4.99 s |

**agg** = server-wide decoded tokens per second; **str** = per-request tokens per second; **TTFT** = time to first token.

Note the ×6 spike: 4.99 s TTFT at 6 concurrent requests — batch efficiency degrades past ×5 under this configuration.

## Files

| File | Purpose |
|---|---|
| `start.sh` | Download model, build/persist SGLang, launch container, wait for readiness |
| `stop.sh` | Confirm and stop/remove the container; cleans up PID + tmp files |

## How it works

1. `start.sh` caches the `Ling-3.0-flash-int4` weights under `$HF_HOME` (uses `hf`, `huggingface-cli`, or falls back to a Docker download).
2. It writes a bootstrap script that, on first run, clones `inclusionAI/sglang_ling_v3` @ `ling_v3_support`, installs Rust + a venv, and `pip install -e`s SGLang — **persisted** in `.sglang-persist/` so subsequent starts skip the ~10 min build.
3. The container runs with `--network host`, `--ipc host`, `--gpus all`, `--shm-size=32g`, and the model snapshot mounted read-only from the HF cache.
4. `start.sh` tails the logs and blocks until `/v1/models` responds, then prints the endpoint.

`.sglang.pid` holds the container ID, `.sglang.log` the launch line; both live in this directory and are removed by `stop.sh`.

## Notes & safety

- **Unified memory**: model weights, compile and KV cache all share host RAM. Keep `MEM_FRACTION_STATIC` at ≤ `0.75` and consider `DOCKER_MEMORY`. If free memory drops below ~10 GB during load, stop the container with `./stop.sh`.
- **No MXFP4**: this repo deliberately uses the INT4 `ling_v3_support` path. The FP4 `flashinfer_mxfp4`/Cutlass path is known-good to OOM/hang this hardware class and is intentionally not supported.
- **Tuning knobs**: after a stable baseline, raise `CTX` gradually, or re-run with `ENABLE_NEXTN=1` to speed up speculative decoding (chipset support varies).

## Troubleshooting

- `curl http://127.0.0.1:8888/v1/models` fails → `docker logs ling-3.0-flash-int4` (first build takes minutes; startup-looking output will stream).
- Container vanished before ready → `./stop.sh` then relaunch with `DOCKER_MEMORY=100g` and see logs.
- Model download stalls → run `./start.sh --download-only --token <HF_TOKEN>` if it is a gated repo.

## License & credits

- Model weights: [inclusionAI/Ling-3.0-flash-int4](https://huggingface.co/inclusionAI/Ling-3.0-flash-int4) (InclusionAI) — check the model card for its license
- SGLang: [sgl-project/sglang](https://github.com/sgl-project/sglang)