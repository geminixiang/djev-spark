"""Raise torch dynamo's recompile limit for the DiffusionGemma sampler.

The compiled sample step is one dynamo specialization per canvas width, and
reads through the structured server arrive in several widths. Past torch's
default of 8 recompiles dynamo runs the function eager for the rest of the
process, which is slower and was where an eager-only dtype mismatch surfaced.
There is no environment variable for the limit, so the raise is written into
the module above the decorator, once, with an anchor that must match exactly.
"""
import importlib.util
import pathlib

site = pathlib.Path(importlib.util.find_spec("vllm").origin).parent
target = site / "model_executor" / "models" / "diffusion_gemma.py"
ANCHOR = "@torch.compile(dynamic=True)\ndef _compiled_sample_step(\n"
MARKER = "# [dgemma-spark] recompile limit"
INSERT = f"""{MARKER}
# Reads come in several canvas widths and each width is a fresh dynamo
# specialization of the sampler. Past torch's default of 8 every later width
# runs eager for the rest of the process.
for _name in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _name):
        setattr(
            torch._dynamo.config,
            _name,
            max(getattr(torch._dynamo.config, _name), 64),
        )


"""

text = target.read_text()
if MARKER in text:
    print("recompile limit already raised")
    raise SystemExit(0)
if text.count(ANCHOR) != 1:
    raise SystemExit(
        f"expected exactly one sample-step decorator in {target.name}, found {text.count(ANCHOR)}; "
        "re-check the anchor after a fork or base bump"
    )
target.write_text(text.replace(ANCHOR, INSERT + ANCHOR, 1))
for pyc in (target.parent / "__pycache__").glob("diffusion_gemma.*.pyc"):
    pyc.unlink()
print("raised the sampler's recompile limit to 64")
