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

The current validation policy combines full numerical comparison, controlled
randomness, FP32 diagnostics and generated-video review; cross-backend bitwise
identity is not required. Four of eleven strict BF16 quality cases retain action
warnings. These warnings remain visible alongside the full FP32 and video
results below. The current strict FA3 + compile profile improves latency by
14.3–14.5% and throughput by 16.7–17.0% against the existing optimized Cosmos
baseline. This is a batch-one, concurrency-one implementation result, with no
scheduler/batching gain established. FA3/compile still require the explicit
experimental option; the following SDPA and earlier FA3 results are historical.

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


### Historical FA3 results before strict joint validation (2026-10-20)

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


### Historical attribution and numerical ablation (2026-10-20)

On the same shared GPU1, current native FA3 + compile + metadata, changing only
request-local UND caching (off/on/off, two warmups and five measured calls per
input per stage) gives median request latency 4.15742 / 3.99242 / 4.14356 s.
Cache-on saves 0.151–0.165 s (3.65–3.97% median latency). This does **not** explain
the earlier entire 16–17% latency reduction. Do not subtract these timings from
the older Cosmos measurements across different load windows. Evidence:
`artifacts/validation/und_ablation_1020_summary.json`.

The two LIBERO outputs pass the normalized gate both against optimized Cosmos
and cache-off, but cache-on/off are not bitwise equal (max-abs 0.03037/0.03946,
RMSE 0.00735/0.00943). Four-domain cache-off ablation reveals a cache-specific
DROID discrepancy: on/off max-abs 0.03862, RMSE 0.01031 fails. Cache-off DROID
also fails the Cosmos RMSE gate (0.010307). RoboCasa, RoboTwin EEF and joint are
bitwise equal on/off, while joint still fails against Cosmos. Both compiled
native execution and caching need numerical fixes; tiny cache unit tests do
not establish full-model equivalence. See `native_fa3_domains_cache_ablation_1020.json`.

Frozen real Q/K/V operator-call probes on GPU0 show the bundled FA3 is faster than NV
FA3 for the tested UND/ACTION/VIDEO shapes. These are isolated FA3 call results (CUDA event intervals can include host
submission gaps),
not raw kernel-only time or a percentage of complete-request gain, and provider outputs differ.
Evidence: `fa3_provider_kernel_profile_1020.json`. Compiler/version and framework
contributions remain unquantified; no scheduler or batching gain is established.

Resource constraint for further task validation: physical GPUs 0–3 only.
Earlier GPU6 results predate this constraint; no new tests may use GPUs 4–7.

Further GPU0 stage diagnostics use one full LIBERO input, two warmups and five
instrumented repetitions, with UND caching disabled on both paths. Reference
worker runtime is 2303.1 ms versus native runtime 2050.4 ms. Network execution
intervals account for 2043.1 versus 1986.9 ms; the remaining 196.5 ms difference
is outside those intervals (preparation/sampling/dispatch). This is a runtime
boundary diagnosis, not a new engine benchmark or an additive breakdown of the
older shared-GPU 19% result. The output still passes the normalized gate
(max-abs 0.036085, RMSE 0.009690). See `request_stage_profile_summary_1020.json`.

A concrete implementation difference is the reference's allocation of a full
41-frame pixel canvas before generic preparation; the native path encodes only
the first observed frame and constructs the latent directly. The reference
already limits VAE encoding to the conditioning prefix, so **full-video VAE
encoding is not the removed work**. Individual preparation operations have not
been separately ablated. Native VAE encode is not faster in this probe
(17.64 ms versus 13.28 ms). These are portable inference-path differences,
not evidence of a vLLM scheduler advantage.

Frozen layer-0/35 replay with identical native code and NV FA3 across Torch
2.13/2.10/2.13 gives roughly 5.75–5.80 ms per layer; actual Cosmos compiled layers
with the exact same fixture inputs give 5.81/5.75 ms. There is no substantial
version or layer-implementation advantage in this limited local probe.
Evidence: `layer_profile_torch{213_before,210,213_after}_1020.json` and
`reference_layer_profile_1020.json`. No claim is made for other shapes or layers.

### Joint numerical and video validation

Cross-backend bitwise identity is not an acceptance requirement. Use
`validate.py --deterministic --save-videos <directory>` with
`PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8` set before startup. Each result
records effective RNG and deterministic settings. External FA3 kernels still
need empirical same-seed repeat checks; PyTorch flags alone do not guarantee
cross-version reproducibility. Independent engines must use distinct
`MASTER_PORT` values as well as distinct allowed `CUDA_VISIBLE_DEVICES`.

Run every input at least twice with the same seed and add matched inputs with
another seed. Compare raw decoded frames, not compressed review MP4 files:

```bash
python tools/pwm_wam/review_video.py \
  --reference artifacts/validation/quality_1020/reference.json \
  --candidate artifacts/validation/quality_1020/native_cached.json \
    artifacts/validation/quality_1020/native_uncached.json \
  --output-dir artifacts/validation/quality_1020/review
```

The report includes active-slot action errors, same-seed repeatability,
cross-backend video RMSE/PSNR/SSIM, temporal-delta differences, and reference
seed variation. A contact sheet and video gallery support visual review.
Similarity metrics are diagnostics, not proof of task success; the tool leaves
visual assessment pending. Existing action thresholds remain visible warnings,
and domain/units/mask/finite-value contracts remain mandatory. Quality runs
include video decoding and are not action-only performance measurements.

#### Additional attribution evidence (2026-10-20)

The reference computes `stable_hash(model.config.action128_spec)` on every
request, including recursive OmegaConf conversion. A same-process reference
experiment on physical GPU3 changed only precomputation of that immutable
fingerprint; batch contract checks remained active. Off/on/off, each with two
warmups and ten measured calls, gave worker medians 2335.65/2250.01/2339.73 ms:
85.64–89.72 ms saved (3.67–3.83%). All 36 action outputs matched. Evidence:
`artifacts/validation/reference_hash_ablation_summary_1020.json` and raw protocol.
This is a portable CPU optimization, not a vLLM scheduler benefit.

A CPU-only preparation ablation replaced zero-padding plus concatenation with
one zero allocation and a copy of the observation. Full canvas tensors and
prepared fields matched, excluding the deliberately fresh UUID output directory.
Five warmups and thirty measurements per phase gave preparation medians
84.51/46.57/84.94 ms, saving 37.93–38.37 ms in that stage. Evidence:
`artifacts/validation/canvas_allocation_ablation_1020.json`.
This does not measure the complete benefit of the native latent-first path and
is not an engine benchmark. Both changes remain diagnostic artifacts; the
Cosmos production baseline was not modified. Neither percentage can be added
to the earlier shared-GPU 19% result or to the separate UND-cache ablation.

#### Joint validation results (2026-10-20)

The quality matrix completed 66 calls: 11 inputs, two repeats, three profiles
(reference, native cached, native uncached), on physical GPUs 0/1/2. Each profile
used batch one and request concurrency one. Same-seed action/raw-video repeats
were stable in all 33 pairs; cache on/off outputs also matched in this strict
configuration. This is observed repeatability, not a cross-backend bitwise gate.
Cross-backend video SSIM ranged from 0.982677 to 0.998688. Sampled keyframes showed
similar scene structure and motion, with local edge/texture differences; this
was not a blind video study or robot success-rate test.

Seven of eleven inputs met the existing action diagnostic thresholds. Four
retained warnings, notably DROID 35-step seed43 with normalized max error 0.181374
and RMSE 0.015455. High video similarity does not erase action warnings. The
corresponding worst normalized point has source-unit error 0.002610; the maximum
across all physical slots is 0.021286, so these must not be confused. Different
slots can have different units. Reference seed-variation controls give context,
not an automatic acceptance threshold. Full evidence and video gallery are in
`artifacts/validation/quality_1020/review/`; detailed conditions, sequence lengths,
limitations and the remaining attribution gap are recorded in the local Cosmos
`plans/v0.3.0e_vllm/report_1019.md`, sections 9–10.

#### Current engineering validation (2026-10-20)

For HTTP numerical/service validation, pass `--deterministic` and set
`PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8` before startup. `/info` exposes
`validation_deterministic`; each action output records effective settings.
The default remains unchanged. Full FP32 diagnostic replay of all four BF16
warning cases matched the independent reference, while BF16 errors remain
reported. The diagnostic loads transformer parameters and RoPE in FP32 and
retains checkpoint BF16 VAE; it is not a production FP32 engine option or a
performance baseline. All 1214 exported tensors matched DCP shape/dtype/value.

The strict-profile twenty-repeat GPU3 A/B/A qualification measured p50
2.370831/2.032670/2.376116 seconds (reference/native/reference), or 14.3–14.5%
lower latency and 16.7–17.0% higher request throughput. There were two complete
LIBERO inputs, each with two warmups and twenty formal calls per profile;
sequences were 6938/6967 tokens, batch one, concurrency one, 5 steps, CFG 1.5,
action-only, 960×640, 40 actions and 41 video frames. Both sides used FA3 +
per-layer compile + metadata reuse. Reference NV FA3 had UND reuse disabled;
native bundled FA3 enabled request-local UND reuse. The legacy numerical gate
still reports a warning for the second input; it was not silently relaxed.

The current HTTP profile passed 1000 mixed-domain isolation requests and
independent process restart replay, plus the original client's video download,
invalid-request recovery, concurrent rejection and disconnect recovery.
The stress cases use smaller images and do not qualify full-size performance.
Regression logs record 50 passed / 9 skipped in the main suite, with all skipped
scopes separately exercised: 6 Hopper FA3 tests and 8 independent reference
tests passed. No full-repository CI claim is made.

Same-GPU controlled attribution established approximately 130 ms from equivalent
reference preparation changes, 68 ms from UND reuse, and 40 ms from the FA3
provider, out of a 344 ms same-GPU ladder difference. Full-runtime environment
A/B/A found only about 4 ms from the environment stack (0.56 ms drift), with
identical observed actions. The remaining approximately 102 ms belongs to the
native execution path, not the scheduler. An equivalent GPU-canvas ablation in
the reference saves about 46 ms; after hash/GPU-canvas preparation, same-GPU,
same-environment, same-provider stage diagnostics leave about 53 ms between
reference worker and native runtime, including about 21 ms in network intervals.
Those diagnostic intervals contain host submission gaps and are not pure kernel
times. The canvas experiment used another allowed GPU, so these stage numbers
corroborate the residual mechanism and scale rather than form an exact additive
partition. The engine wrapper itself shows no scheduling speedup.

The tested preset completes **engineering acceptance with disclosed BF16
numerical warnings**, not robot closed-loop qualification. Machine-readable
results are in `artifacts/validation/acceptance_1020/final_acceptance.json`, with
evidence hashes, profile limits, the four warning inputs and final attribution.
Earlier failed summaries retain their original status and link to this separate
joint assessment. Commands/raw outputs are preserved in the same directory.
The local Cosmos `plans/v0.3.0e_vllm/report_1019.md` contains full conditions and
source references; `quality_1020/review/index.html` contains the video gallery.
The gains principally reflect portable preparation, request-local reuse,
provider and specialized execution changes. Porting those optimizations to
Cosmos would narrow this comparison; true vLLM scheduling/batching advantages
require a separate implementation and controlled comparison.
