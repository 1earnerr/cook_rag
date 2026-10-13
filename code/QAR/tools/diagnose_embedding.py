"""Layered, timeout-bounded diagnosis for the configured BGE embedding stack.

This script does not import or modify QAR application code. Each layer runs in a
fresh Python process so a native Torch/Transformers failure cannot hide the
result of earlier layers.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import textwrap
from pathlib import Path


MODEL = "BAAI/bge-small-zh-v1.5"
TIMEOUT_SECONDS = int(os.getenv("QAR_EMBED_DIAGNOSTIC_TIMEOUT", "180"))

COMMON = r'''
import os, sys, platform
print("python", sys.version.replace("\n", " "), flush=True)
print("executable", sys.executable, flush=True)
print("HF_HOME", os.getenv("HF_HOME"), flush=True)
print("HUGGINGFACE_HUB_CACHE", os.getenv("HUGGINGFACE_HUB_CACHE"), flush=True)
try:
    from huggingface_hub import constants as hf_constants
    print("actual_hf_home", hf_constants.HF_HOME, flush=True)
    print("actual_hf_hub_cache", hf_constants.HF_HUB_CACHE, flush=True)
    print("hf_hub_offline", hf_constants.HF_HUB_OFFLINE, flush=True)
except Exception as exc:
    print("huggingface_cache_environment_error", repr(exc), flush=True)
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), flush=True)
    print("torch_cuda", getattr(torch.version, "cuda", None), flush=True)
except Exception as exc:
    print("torch_environment_error", repr(exc), flush=True)
'''

STEPS = [
    ("imports", COMMON + r'''
import transformers, sentence_transformers
print("transformers", transformers.__version__, flush=True)
print("sentence_transformers", sentence_transformers.__version__, flush=True)
'''),
    ("tokenizer", COMMON + f'''\nfrom transformers import AutoTokenizer\nprint("start tokenizer", flush=True)\ntokenizer = AutoTokenizer.from_pretrained({MODEL!r})\nprint("tokenizer vocab", tokenizer.vocab_size, "model_max_length", tokenizer.model_max_length, flush=True)\n'''),
    ("automodel", COMMON + f'''\nfrom transformers import AutoModel\nprint("start automodel", flush=True)\ntokenizer = None\nmodel = AutoModel.from_pretrained({MODEL!r})\nprint("model config max_position_embeddings", getattr(model.config, "max_position_embeddings", None), flush=True)\n'''),
    ("sentence_transformer", COMMON + f'''\nfrom sentence_transformers import SentenceTransformer\nprint("start sentence_transformer", flush=True)\nmodel = SentenceTransformer({MODEL!r}, device="cpu")\nprint("sentence_transformer max_seq_length", model.max_seq_length, flush=True)\n'''),
    ("langchain_embeddings", COMMON + f'''\nfrom langchain_huggingface import HuggingFaceEmbeddings\nprint("start langchain_embeddings", flush=True)\nembeddings = HuggingFaceEmbeddings(model_name={MODEL!r}, model_kwargs={{"device": "cpu"}}, encode_kwargs={{"normalize_embeddings": True}})\nprint("langchain_embeddings initialized", flush=True)\n'''),
    ("embed_query", COMMON + f'''\nfrom langchain_huggingface import HuggingFaceEmbeddings\nprint("start embed_query", flush=True)\nembeddings = HuggingFaceEmbeddings(model_name={MODEL!r}, model_kwargs={{"device": "cpu"}}, encode_kwargs={{"normalize_embeddings": True}})\nvector = embeddings.embed_query("这是一个测试问题")\nprint("query_dimension", len(vector), "finite", all(abs(x) < float("inf") for x in vector), flush=True)\n'''),
    ("embed_documents", COMMON + f'''\nfrom langchain_huggingface import HuggingFaceEmbeddings\nprint("start embed_documents", flush=True)\nembeddings = HuggingFaceEmbeddings(model_name={MODEL!r}, model_kwargs={{"device": "cpu"}}, encode_kwargs={{"normalize_embeddings": True}})\nvectors = embeddings.embed_documents(["这是第一段测试文本。", "这是第二段测试文本。\"])\nprint("document_count", len(vectors), "dimensions", [len(v) for v in vectors], flush=True)\n'''),
]


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def run_step(name: str, code: str) -> None:
    print(f"\n[{name}] start={now()} timeout={TIMEOUT_SECONDS}s", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        print(f"status=TIMEOUT finish={now()}", flush=True)
        output = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        print(output, end="", flush=True)
        return
    print(f"status={'SUCCESS' if result.returncode == 0 else 'FAILED'} returncode={result.returncode} finish={now()}", flush=True)
    if result.stdout:
        print("stdout:", result.stdout, end="", flush=True)
    if result.stderr:
        print("stderr:", result.stderr, end="", flush=True)


def main() -> None:
    print("model", MODEL)
    print("diagnostic_script", Path(__file__).resolve())
    for name, code in STEPS:
        run_step(name, textwrap.dedent(code))


if __name__ == "__main__":
    main()
