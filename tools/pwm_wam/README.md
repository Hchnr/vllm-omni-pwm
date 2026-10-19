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
