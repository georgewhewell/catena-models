# LFM3 d1 System One decisions

`LiquidAI/d1-2B-GGUF:BF16` answers typed questions without generating tokens.
CLI and HTTP use the same JSON request and response. The language checkpoint and
its own `mmproj-d1-2B-BF16.gguf` projector are selected from the Hugging Face cache.

Save this as `request.json`:

```json
{
  "state": "I was charged twice this month, please refund one of them.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Charges, refunds, invoices",
        "technical": "App or site faults",
        "fraud": "Suspected unauthorised use"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["Can wait", "Today", "Blocking the customer now"]
    }
  }
}
```

```sh
cargo run -r -- --model LiquidAI/d1-2B-GGUF:BF16 --systemone request.json
cat request.json | cargo run -r -- --model LiquidAI/d1-2B-GGUF:BF16 --systemone -

cargo run -r -- --model LiquidAI/d1-2B-GGUF:BF16 --serve --ctx-size 8192
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
```

CLI stdout contains one JSON response; loading and compilation diagnostics go to
stderr. Existing server bearer authentication and queue limits apply. An optional
request `model` must match `--model`.

The response contains `model`, named `answers`, and
`usage: {"input_tokens": N, "output_tokens": 0}`. Every question reads a complete
prompt, so input usage includes its repeated state and expanded image tokens.

- `noul` returns `{"type":"noul","noul": P(yes)}`. Optional criteria describe
  `"true"` and `"false"`.
- `choice` accepts 1–255 named options and returns `choice`, `confidence`, and
  `probabilities`. Option order is preserved; descriptions may be null.
- `score` accepts 2–10 levels, lowest first, and returns the **median level**,
  `confidence`, `probabilities`, and `legend`. This follows llama.cpp's LFM3 d1
  behavior. Confidence is the largest option probability.

`state` is required and can be text or JSON, or null for media-only questions.
Add PNG/JPEG base64 data URLs through `images` or its `files` alias. Up to eight
images are supported, in order. Image parts in state chat messages are accepted
as well: `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`.
The server automatically loads d1's projector; CLI loads it for image requests.
Images are encoded once per request and reused across its questions. HTTP requests
retain the existing 8 MiB body limit. External URLs, audio, video, multiple choice,
extraction, numeric estimates, boxes, streaming, and Python-only readout switches
are unsupported.

## Execution and compatibility

The existing LFM3 weights, normalization, projections, RoPE, convolution, vision
encoder and image insertion kernels are reused. d1's full-attention layers read
all prompt tokens bidirectionally, while sliding attention and convolution remain
causal. Both matrix-core and SIMT attention specialize the key traversal, padding
mask and value-reduction bounds. Each question starts from fresh prompt-sized
scratch state. Decisions do not use chat prefix caching, chunked prefill,
continuation, or generation APIs.

The GGUF supplies `lfm3.decision.type`, `lfm3.attention.causal`,
`tokenizer.chat_template.systemone`, and per-type calibration temperatures.
The named template defines decision inputs; the ordinary chat template is for
conversation. Selected vocabulary token forms are max-pooled per option, divided
by the checkpoint temperature, and softmaxed. No additional learned head or Python
runtime is needed. The original model's `prompt.py`, `api.py`, `calibration.py`, and
`attention.py` explain the reference prompt and readout. Python's packed question
sets and broader typed API are outside this implementation.

See the [GGUF model card](https://huggingface.co/LiquidAI/d1-2B-GGUF) for llama.cpp
examples. Errors use the existing JSON error envelope: invalid requests return
400, and incompatible model capabilities return 501. d1 rejects Chat Completions
and Responses text generation; generative models reject `/v1/systemone`.

## Validation

Focused CPU tests run with the normal library/server suite. Cached metadata,
label aliases, exact template rendering and HIP/CUDA graph compilation:

```sh
cargo test -r --lib systemone::tests::cached_metadata_template_and_graphs -- --ignored --nocapture
```

GPU integration covers text, JSON, a prompt beyond the sliding window, and
ordered red/blue images with null state:

```sh
cargo test -r --lib systemone::tests::decisions_match_reference -- --ignored --nocapture
D1_SIMT=1 cargo test -r --lib systemone::tests::decisions_match_reference -- --ignored --nocapture
```

`D1_TEST_OUTPUT=/path` saves the test requests and responses. Run those requests
against a llama.cpp d1 server and save its responses as a JSON array; pass that
file as `D1_REFERENCE=/path/reference.json` to check usage, selected decisions and
calibrated probabilities (absolute tolerance 0.02). `CATENA_GPU_DIALECT=cuda`
selects CUDA; hardware validation must be run separately on NVIDIA hardware.

On HIP/Strix Halo, both matrix-core and SIMT execution passed these four fixtures
against llama.cpp with the same BF16 checkpoint. Maximum absolute probability
differences were 0.01365 and 0.01503 respectively; input-token usage and selected
answers matched. These are focused numerical checks, not a model-quality study.

The complete current decision language graph also passed HIP/CUDA Catena code
generation and nvcc sm_80 PTX compilation with native BF16 MMA present. NVIDIA
execution remains unverified. CLI file/stdin and live HTTP responses matched
exactly, and the existing LFM3-VL image/decode/checkpoint regression passed.
