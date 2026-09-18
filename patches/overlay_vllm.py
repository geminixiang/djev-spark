"""Copy the fork's changed vllm/ files over the base image's vLLM.

usage: overlay_vllm.py <fork checkout> <base commit>

The checkout carries changed.txt, the paths under vllm/ that differ from the
base commit. The base image's vLLM must report that commit in its version and
every target file must already exist, or the build stops: an overlay onto a
different vLLM would import and then fail in ways that look like model bugs.
"""
import importlib.metadata
import importlib.util
import pathlib
import shutil
import sys

fork = pathlib.Path(sys.argv[1])
base = sys.argv[2]

version = importlib.metadata.version("vllm")
short = base[:9]
if f"+g{short}" not in version:
    raise SystemExit(
        f"base image vLLM is {version}; the overlay expects commit {short}. "
        "Bump VLLM_BASE and VLLM_REF together, against a fork branch built on that commit."
    )

site = pathlib.Path(importlib.util.find_spec("vllm").origin).parent
changed = [line.strip() for line in (fork / "changed.txt").read_text().splitlines() if line.strip()]
if not changed:
    raise SystemExit("changed.txt is empty; the fork ref carries no vllm/ changes")

for rel in changed:
    src = fork / rel
    dst = site / pathlib.Path(rel).relative_to("vllm")
    if not dst.exists():
        raise SystemExit(f"{dst} is not in the base image; the base moved")
    shutil.copyfile(src, dst)
    for pyc in (dst.parent / "__pycache__").glob(dst.stem + ".*.pyc"):
        pyc.unlink()
    print("overlay", rel)
print(f"overlaid {len(changed)} files onto vllm {version}")
