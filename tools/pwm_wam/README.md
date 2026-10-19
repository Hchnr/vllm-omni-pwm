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

Input is the existing Cosmos WAM request JSON with
`extra.action128.return_video=false`. Supply it as
`OmniDiffusionSamplingParams.extra_args["pwm_request"]`; the sampling parameters'
seed, steps and guidance take precedence. The output payload carries normalized
and physical Action128 values, sparse mask, source domain and timestamps.

Only single-GPU non-AR action delivery is currently implemented. The video tower
still participates in joint denoising. Generic caching, step batching and
parallelism are not advertised. UND caching is experimental and disabled by
default pending full-trajectory validation.
