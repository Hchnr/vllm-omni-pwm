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

Against the eager SDPA numerical reference, sixteen cases match **exactly** in normalized and
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

Deployment acceptance is **limited to the tested LIBERO profiles**. The baseline
is Cosmos FA3 + per-layer compile + request-local metadata reuse, with HTTP CUDA
Graph, UND compute caching and QKV fusion disabled. Native bundled FA3 + compile
+ raw UND projection caching passes the two LIBERO inputs at 5 and 35 steps.
On shared-GPU A/B/A, the 5-step profile improves median latency by 16–17% and
throughput by 19% against that optimized baseline. Cross-domain correctness is
not complete: DROID fails the locked tolerance. Both FA3 modes remain explicitly
experimental. Eager SDPA is a numerical reference only.

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
fix; that historical failure must be retested at the current revision. The production defaults reject FA3 and compilation unless the explicit
`pwm_experimental_backend` model option is enabled for validation.
Quantization, LoRA, generic diffusion caches, step batching, TP/CP, offload,
sleep mode, autoregression and state/action conditioning are unsupported and
rejected. The standalone `validate_native.py` retains backend/compile options
for numerical investigations; those options do not imply production support.
FA3 has a fullgraph-compatible custom op and request-local cumulative sequence
metadata. Compiled UND reuse is covered by Hopper layer-level tests; complete
trajectory and deployment acceptance remains a separate requirement.

## Independent reference tools

`validate.py --mode worker` and `capture_native_fixture.py` use the separate
Cosmos environment to load the original DCP/HF model. Captured tensors use
safetensors and are only validation inputs. `replay_native_fixture.py`,
`replay_native_network.py` and `compare_native_trace.py` isolate layer, full
network and sampling-boundary errors. `audit_hf.py` checks export structure.

The earlier `PwmWamPipeline` compatibility adapter is retained for reference
experiments. It uses `pwm_python`/`pwm_cosmos_root` and runs a Cosmos subprocess;
it is not the native implementation. Its projected-UND/compile experiments
must not be used as native-engine deployment acceptance evidence.

### Optimized deployment comparison

Use `validate.py --mode worker --attention-backend flash3 --compile` for the
existing Cosmos baseline (metadata reuse is enabled by the worker). Keep
checkpoint, request JSON/assets, seed, steps, CFG, output mode, warmup and repeats
fixed. Run baseline/candidate/baseline on the same physical GPU and inspect
co-tenant load. `compare.py` reports deployment eligibility separately from
numerical correctness; eager SDPA cannot qualify as this baseline.

`validate.py --mode runtime --native` measures experimental native backend and
compiler combinations with the same timing/provenance harness. It bypasses the
engine's production configuration guard for investigation only. These results
are not an end-to-end vLLM deployment acceptance; final measurements must use
`--mode engine --native`. Different attention implementations are permitted in
a performance comparison, but numerical tolerances and workload remain fixed.

The 2026-10-20 corrected shared-GPU A/B/A showed the existing SDPA native
configuration was slower: two inputs had p50 6.63/7.32 seconds versus
4.60/4.63 for Cosmos FA3+compile. One SDPA candidate trajectory also failed
the numerical tolerance against this optimized reference. The earlier eager
SDPA speedup is therefore not a deployment win. Raw corrected measurements:
`artifacts/validation/optimized_baseline_1020_{reference,sdpa_candidate,after_reference}.json`.

For experimental native FA3+compile engine measurements, add
`--experimental-native-backend --attention-backend flash3 --compile` to
`validate.py --mode engine --native`. `--native-und-cache` tests compiled
request-local UND reuse. Keep these opt-in until full-trajectory validation and
the optimized-reference performance checks pass.

### Match the reference FA3 provider

`flash3` uses vLLM's bundled FA3. `flash3_nv` uses the optional
`flash_attn_3_nv` package, matching the existing Cosmos kernel implementation.
The two providers can produce different BF16 outputs on identical noncausal
Q/K/V; they must be validated separately over full trajectories. The native
model never imports Cosmos for either provider.

For local reproducibility, the reference environment's
`flash_attn_3_nv==1.0.3+cu130.torch210` wheel was repacked without changing its
binaries and installed with `uv pip install --python .venv/bin/python --no-deps`
into the native PyTorch 2.13 environment. The local wheel and SHA256 provenance
are under `artifacts/validation/nv_fa3_install_provenance_1020.json`. This records
the tested environment, not general cross-version ABI compatibility. Install
the matching optional dependency before selecting `flash3_nv`.

Compiled requests scope Dynamo's recompile/accumulated limits to 512/4096,
matching the reference capacity, and restore the process settings afterwards.
The default limit of eight fails after a few shapes because CFG branches and
cache fill/reuse specialize separately. This does not promise zero cold
compilation for new shapes.

HTTP validation supports `--attention-backend flash3_nv --compile --cache-und
--experimental-backend`; `/info` exposes all four selections. Production defaults
remain unchanged because expanded optimized-reference validation has failures.


### Corrected measured results (2026-10-20)

Two complete LIBERO requests, 960×640, 40 actions / 2 seconds, 5 denoising steps,
CFG 1.5, BF16, seeds 42/43, actions only; each stage has 2 warmups and 20 measured
runs per request on the same shared H100 GPU1:

| Path | Input 0 p50 / p95 (s) | Input 1 p50 / p95 (s) | chunks/s |
| --- | --- | --- | --- |
| Cosmos FA3 + compile, before | 4.6209 / 4.6443 | 4.6324 / 4.6600 | 0.22397 |
| Native FA3 + compile + UND cache | 3.8529 / 3.8686 | 3.8446 / 3.8685 | 0.26760 |
| Cosmos FA3 + compile, after | 4.6002 / 4.6170 | 4.5915 / 4.6268 | 0.22508 |

All 40 candidate outputs pass both references at normalized active-slot
max-abs ≤ 0.05 and RMSE ≤ 0.01. Evidence:
`artifacts/validation/final_fa3_ab_1020_{reference_before,candidate,reference_after,comparison_before,comparison_after}.json`.
This measures the integrated implementation, not an isolated scheduler benefit.
The existing result does not quantify individual contributions: UND caching is
a plausible source, but a same-revision cache-off/on/off ablation is required
before attributing the measured gain to it. UND caching can also be implemented
in Cosmos. There is no real batching or
continuous-batching throughput result. Startup remains slower (128 s versus
53 s, excluding request warmups). Shared co-tenant GPU load limits generalization.

Expanded bundled-FA3 testing fails DROID (max-abs 0.05495 / RMSE 0.01113),
while RoboCasa and RoboTwin EEF pass. The final rerun completes all four domain
requests without a compilation-limit error, but RoboTwin joint fails RMSE
(0.01275). Across four LIBERO and four additional domain trajectories, six of
eight pass. Evidence: `native_engine_fa3_domains_final_comparison_1020.json`.
Switching to the reference's NV FA3 does not solve
compiled rounding differences: five of eight NV trajectories fail, including
RoboTwin EEF/joint. Failed NV runs are not performance evidence. Layer and
attention fixtures localize differences in compiled UND K norm/RoPE, but the
root cause is not yet fixed. Diagnostic variants remain untracked artifacts;
none changed production numerical behavior or relaxed the tolerance.

Regression evidence: `final_native_suite_1020.log` (51 passed, 3 skipped) and
`independent_reference_tests_1020.log` (8 passed in the Cosmos environment).
The earlier 16-case exact SDPA and HTTP stress checks do not establish FA3
cross-domain correctness or FA3 service stress reliability.

The FA3/compile/cache HTTP profile passes main-thread warmup to HTTP-thread
execution, A/B/A replay, exact equality with direct-engine actions, recovery
after invalid requests, and the unmodified Cosmos `ray.submit` client.
Evidence: `artifacts/validation/fa3_1020_http_validation.json`. This is not a
FA3 video or long-duration stress test. The validation server was shut down.
The machine-readable corrected summary is `optimized_acceptance_1020.json`;
the old `final_native_acceptance.json` is marked as historical SDPA-only evidence.
