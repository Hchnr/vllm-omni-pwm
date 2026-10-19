# PWM WAM integration validation

`validate.py` runs actual requests through either the isolated reference runtime
or the vLLM-Omni diffusion engine. Use the same checkpoint, precision, inputs,
steps, guidance and GPU for timing comparisons. Warmup is excluded; retain all
outputs to check correctness before interpreting performance.

This is a compatibility backend, not a port of the three-tower transformer to
upstream Cosmos3 kernels. It requires an installed Cosmos PWM environment via
`pwm_python` and `pwm_cosmos_root` in `OmniDiffusionConfig.model_config`. Select
`model_class_name="PwmWamPipeline"` explicitly. Do not add a Diffusers
`model_index.json` to the native HF artifact: that changes the Cosmos loader's
weight mapping. The artifact consists of `config.json`, a safetensors index and
shards exported by Cosmos `scripts.export_model --no-vit --no-use-ema-weights`.

Input is the existing Cosmos WAM request JSON. Set
`extra.action128.return_video=false` for action-only latency measurements. Supply it as
`OmniDiffusionSamplingParams.extra_args["pwm_request"]`; the sampling parameters'
seed, steps and guidance take precedence. The output payload carries normalized
and physical Action128 values, sparse mask, source domain and timestamps.

The single-GPU non-AR path always runs the video tower during joint denoising.
With `return_video=true`, it also decodes frames and returns a uint8 THWC array
in the Omni output's `images` list. Frames use a private temporary NumPy file
between environments (no pickle), removed after delivery.

Generic caching, step batching, parallelism, sleep and offload are unsupported.
UND caching is opt-in: enable both `pwm_reuse_und` and `pwm_projected_und` to use
the projected-key cache. The legacy normalized-key cache fails the BF16
compiled acceptance test and must not be treated as the validated optimization.

## Native worker implementation (validation in progress)

`PwmWamNativePipeline` runs the three towers, Wan VAE, tokenizer, and UniPC in
one Omni worker process. It does not import Cosmos or spawn its runtime.
Use `model_class_name="PwmWamNativePipeline"` and model config fields
`pwm_vae_path`, `pwm_tokenizer_path`, `pwm_attention_backend` (`sdpa` or
`flash3`), and optional `pwm_compile` (default false). Set `enforce_eager=True`
on Omni: the optional PWM compile switch controls only transformer layers.
Native request schema is the same `extra_args.pwm_request` envelope as above;
conditioning currently accepts a local image. One GPU, one request per forward.
Sleep/offload, generic cache backends, quantization, and LoRA are rejected.

Full HF tensor coverage and real-request 36-layer velocity fixture replay are
exact with SDPA. The VAE adapter preserves the checkpoint's BF16 RMSNorm
arithmetic; newer Diffusers otherwise upcasts that normalization and changes
conditioning. End-to-end acceptance and performance gates remain in progress;
FA3 and compile are explicit experimental choices until those gates pass.

Standalone native validation (Omni environment only):

```bash
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH='' .venv/bin/python tools/pwm_wam/validate_native.py \
  --checkpoint artifacts/pwm-wam-hf \
  --vae /path/to/Cosmos3-Nano/vae --tokenizer /path/to/Cosmos3-Nano-VLM \
  --input /path/to/request.json --output /path/to/native_result.json
```

For actual Omni engine verification use `validate.py --mode engine --native`
with `--vae`, `--tokenizer`, and `--attention-backend sdpa`. The legacy
`--cosmos-root` and `--python` arguments still populate provenance; native
execution does not use that interpreter. `capture_native_fixture.py` is
reference-only tooling and must run in the Cosmos environment. Its safetensors
fixtures are validation artifacts, not production inputs.

### Native HTTP facade

The optional local facade uses the native Omni worker and accepts the existing
Cosmos `ray.submit` `/generate` request and `SampleOutputs` response envelope.
Run `python -m vllm_omni.diffusion.models.pwm_wam.serving --help` for checkpoint,
VAE, tokenizer, output directory and warmup flags. It defaults to loopback,
BF16 eager SDPA and one request at a time (503 + Retry-After when busy).
Conditioning images must be local files accessible to the worker. Video files
are saved under unique server-generated directories and served through
`/outputs`; `/info` exposes configuration/contract hashes and warmup status.
GPU integration and original-client acceptance are tracked separately from
model-free HTTP tests. Do not infer benchmark acceptance from HTTP unit tests.

Native acceptance update: BF16 eager SDPA matches the original eager DCP
reference exactly for two full five-step action trajectories and all ten
conditional/unconditional network boundaries of the second request. Native
FA3 failed the final action tolerance (second input max-abs 0.14824, RMSE
0.02445); the production native pipeline now rejects FA3 and compilation.
The standalone replay/runtime tools retain them only for investigations.
`--native-und-cache` enables an experimental request-local UND cache in engine
validation. It is off by default and cannot be combined with compilation.
