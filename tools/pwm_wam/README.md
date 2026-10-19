# PWM WAM native integration

`PwmWamNativePipeline` runs the Action128 three-tower transformer, Wan VAE,
tokenizer and UniPC directly in the vLLM-Omni diffusion worker. Production
execution has no Cosmos import or reference-runtime subprocess. The supported
configuration is Nano `width_only_1b`, BF16 eager SDPA, no-state WAM, first-frame
image conditioning, one GPU and one request at a time.

The native HF artifact contains `config.json`, `model.safetensors.index.json`
and all shards. **Do not add `model_index.json`**: it activates a different
Diffusers weight mapping. The loader checks every native tensor name and shape,
keeps separate source-normalization and projection IDs, and checks the stored
Action128 contract digest. VAE and tokenizer paths must be locally available.

## Run the native HTTP service

```bash
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH='' OMP_NUM_THREADS=1 \
  .venv/bin/python -m vllm_omni.diffusion.models.pwm_wam.serving \
  --checkpoint artifacts/pwm-wam-hf \
  --vae /path/to/Cosmos3-Nano/vae \
  --tokenizer /path/to/Cosmos3-Nano-VLM \
  --output-dir outputs/pwm-wam --warmup-input /path/to/request.json \
  --cache-und --host 127.0.0.1 --port 8019
```

This accepts the existing Cosmos `ray.submit` client through `/generate` and
returns the original `SampleOutputs` envelope. `/outputs` serves generated
files; `/info` reports configuration/contract hashes and completed warmup;
`/models`, `/domains`, and `/healthz` support discovery. Conditioning images
must be local files accessible to the worker. Concurrent generation returns
503 with `Retry-After`; a disconnected caller does not release model ownership
before its worker request finishes. Use one server process per GPU.

`--cache-und` reuses UND hidden states and K/V within one request, separately
for each CFG branch. It defaults to off. Nothing is cached across requests.
The VIDEO and ACTION towers still run jointly at every denoising step.
`extra.action128.return_video=false` omits VAE decoding and video delivery,
while `true` writes an MP4 that the original client downloads.

## Direct engine configuration and validation

Select `model_class_name="PwmWamNativePipeline"`, `enforce_eager=True`,
`max_num_seqs=1`, and model config fields `pwm_vae_path`,
`pwm_tokenizer_path`, `pwm_attention_backend="sdpa"`, and optionally
`pwm_native_und_cache=True`. Supply the existing WAM request JSON as
`OmniDiffusionSamplingParams.extra_args["pwm_request"]`. Sampling parameters'
seed, steps and guidance take precedence over the embedded JSON.

```bash
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH='' OMP_NUM_THREADS=1 \
  .venv/bin/python tools/pwm_wam/validate.py \
  --mode engine --native --native-und-cache \
  --checkpoint artifacts/pwm-wam-hf \
  --vae /path/to/Cosmos3-Nano/vae --tokenizer /path/to/Cosmos3-Nano-VLM \
  --cosmos-root /path/to/cosmos-framework-pwm \
  --python /path/to/cosmos-framework-pwm/.venv/bin/python \
  --attention-backend sdpa --input /path/to/request.json \
  --warmup 2 --repeats 20 --check-invalid-request --output /path/to/native.json
```

The legacy `--cosmos-root` and `--python` arguments in this validation tool
record reference provenance; native execution does not invoke that interpreter.
Without `--steps`, the tool preserves each JSON's step count. It records every
output and latency, per-input p50/p95, image hashes, revisions and GPU process
snapshots. Compare with `compare.py`: normalized active-slot max-abs must be
at most 0.05 and RMSE at most 0.01; inactive slots must be exactly zero.
Matching benchmark arguments do not establish GPU exclusivity. Inspect the
recorded co-tenant processes before interpreting timing.

## Numerical acceptance and limits

Sixteen independent-reference cases now match **exactly** in normalized and
physical actions: real LIBERO/DROID/RoboCasa/RoboTwin observations, 12 new fixed seeds
plus two original seeds, one/two/three camera layouts, CFG 1/1.5, 5/10/35 steps,
short/two-second horizons and different video/action FPS. Reduced-resolution
cases test the numerical contract; they do not establish robot task quality.
The matrix includes native request-local UND caching, five source domains,
projection IDs 0/2 and 10/14/20 active action dimensions. Original-client action,
video download/decode, warmup-to-HTTP-thread and A/B/A/restart replay also pass.
Two 1,000-request small-resolution stress runs (cache off/on) had exact
replay hashes. A real HTTP disconnect retained model ownership until completion
and recovered with exact actions.

Performance acceptance is **not complete**. The deployment baseline is Cosmos
FA3 + per-layer compile + request-local metadata reuse, with HTTP CUDA Graph,
UND compute caching and QKV fusion disabled. Native superiority over that
baseline has not been established. Eager SDPA is a numerical reference only.

On the task's shared H100 80GB, same-GPU A/B/A against eager SDPA (two inputs,
two warmups and twenty measurements per input per stage), native UND caching
reduced each input's p50 by 8.8–9.7% and increased aggregate hot-request
throughput by 9.6–10.6%. All measured outputs matched exactly. Co-tenant activity
was recorded; these are not exclusive-GPU numbers or comparisons against the
existing optimized Cosmos deployment. Startup was slower (about 129 seconds
versus 52 seconds for the first reference stage); prefer persistent serving.
Raw results are in `artifacts/validation/final_native_acceptance.json` and
`native_cache_ab_{reference_before,candidate,reference_after}.json` locally.

The compatibility arithmetic is deliberate: explicit SDPA masks, FP32 time
embeddings, original BF16 VAE RMS normalization, TF32 disabled, cuDNN SDPA
enabled, CPU sigma coefficients and the original UniPC 2x2 corrector rounding.
A frozen PyTorch 2.10 CUDA fixture checks complete solver trajectories from a
different runtime version. Same-environment reference-source tests alone did
not catch the corrector rounding difference.

An earlier native FA3 experiment failed full-trajectory tolerance (second
initial input max-abs 0.14824, RMSE 0.02445), before the final UniPC rounding
fix; that historical failure must be retested at the current revision. The production pipeline rejects FA3 and compilation.
Quantization, LoRA, generic diffusion caches, step batching, TP/CP, offload,
sleep mode, autoregression and state/action conditioning are unsupported and
rejected. The standalone `validate_native.py` retains backend/compile options
for numerical investigations; those options do not imply production support.

## Independent reference tools

`validate.py --mode worker` and `capture_native_fixture.py` use the separate
Cosmos environment to load the original DCP/HF model. Captured tensors use
safetensors and are only validation inputs. `replay_native_fixture.py`,
`replay_native_network.py` and `compare_native_trace.py` isolate layer, full
network and sampling-boundary errors. `audit_hf.py` checks export structure.

The earlier `PwmWamPipeline` compatibility adapter is retained for reference
experiments. It uses `pwm_python`/`pwm_cosmos_root` and runs a Cosmos subprocess;
it is not the native implementation. Its projected-UND/compile experiments
must not be used as acceptance evidence against the stricter eager baseline.
