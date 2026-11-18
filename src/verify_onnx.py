# -*- coding: utf-8 -*-
"""导出模型的数值一致性验证：ONNX 输出 vs 框架输出。

【这份脚本回答什么问题】
    导出后的模型**是不是同一个模型**。它不做业务评测、不挑阈值、不报准确率——
    只回答"同样的输入，两个实现给出的 logits 一样吗"。把这件事单独隔离出来，
    才能在出问题时区分"模型/导出错了"和"上游数据错了"。

【为什么必须单独验一次，而不是直接看下游指标】
    下游指标会掩盖导出错误：分类边界附近的样本本来就会翻，指标掉 1~2 个点看不出原因。
    逐条比对 logits 则能把错误定位到"这个文件坏了"，而且偏差量级直接告诉你坏在哪一层：
      · 1e-6 ~ 1e-5  —— 浮点累加顺序差异，正常；
      · 1e-3 量级    —— 导出/算子替换有问题；
      · 0.1 以上     —— 拿错文件、类别顺序错位、或分词结果不同。

【判定标准不写死任何指标数字】
    全部由随包的参考数据推出：偏差必须在 ``--tol`` 内，逐条 argmax 一致率必须达到
    ``--min-agreement``（默认 100%）。换模型后不需要改脚本。

【--demo 做什么】
    不需要模型文件：用固定种子造一批输入与"参考 logits"，再跑一遍比较流程，
    用来演示报告格式与通过/不通过逻辑。它**不验证任何真实模型**，输出会明确标注 synthetic。

用法：
    python -m src.verify_onnx --onnx model.onnx --data reference.npz
    python -m src.verify_onnx --demo                       # 只看流程，不需模型
    python -m src.verify_onnx --demo --json out.json

参考数据（.npz）需要这些键：
    input_ids        (N, L) int   —— 已按导出时的分词规则切好并 padding 到固定长度 L
    attention_mask   (N, L) int
    ref_logits       (N, C) float —— 框架实现（导出源）在同一批输入上的输出
可选：
    labels           (N,)         —— 只用于在报告里按类分组打印，不参与判定
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

try:                                  # numpy 是硬依赖，但 --help 不该因为缺它而崩
    import numpy as np
except ImportError:                   # pragma: no cover
    np = None


# ---------------------------------------------------------------- 核心：比对


def compare(ref_logits, got_logits, tol: float = 1e-3) -> dict:
    """逐条比对两组 logits，返回偏差、一致率与不一致的下标。"""
    ref = np.asarray(ref_logits)
    got = np.asarray(got_logits)
    if ref.shape != got.shape:
        raise ValueError(f"形状不一致：参考 {ref.shape} vs 本次 {got.shape}")

    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    per_item_max = diff.reshape(len(ref), -1).max(axis=1)
    ref_arg = ref.argmax(axis=1)
    got_arg = got.argmax(axis=1)
    mismatch = [int(i) for i in np.nonzero(ref_arg != got_arg)[0]]

    return {
        "items": int(len(ref)),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "worst_items": [int(i) for i in np.argsort(per_item_max)[::-1][:5]],
        "agreement": int(len(ref) - len(mismatch)),
        "agreement_rate": float(1.0 - len(mismatch) / len(ref)) if len(ref) else 1.0,
        "mismatch": mismatch,
        "within_tol": bool(diff.max() <= tol) if diff.size else True,
    }


def describe_deviation(max_abs: float) -> str:
    """把偏差量级翻译成"可能坏在哪一层"，让人不用猜。"""
    if max_abs <= 1e-5:
        return "float noise only - export looks lossless"
    if max_abs <= 1e-3:
        return "small but above pure float noise - check operator substitution"
    if max_abs <= 1e-1:
        return "large - suspect wrong file, class-order mismatch or different tokenization"
    return "huge - the two sides are not the same model"


# ---------------------------------------------------------------- 真机：跑 ONNX


def inspect_model(sess, expect_len: int | None = None) -> dict:
    """打印/检查模型的输入输出契约。静态形状与类别数是边缘部署的硬约束。"""
    inputs = [{"name": i.name, "shape": list(i.shape), "type": i.type} for i in sess.get_inputs()]
    outputs = [{"name": o.name, "shape": list(o.shape), "type": o.type} for o in sess.get_outputs()]
    dynamic = [i["name"] for i in inputs
               if any(isinstance(d, str) or d is None for d in i["shape"])]
    problems = []
    if dynamic:
        problems.append(f"dynamic input shape on {dynamic} - edge accelerators need static shapes")
    if expect_len is not None:
        fixed = [i for i in inputs if i["shape"] and isinstance(i["shape"][-1], int)]
        wrong = [i["name"] for i in fixed if i["shape"][-1] != expect_len]
        if wrong:
            problems.append(f"input length != reference length {expect_len}: {wrong}")
    classes = None
    if outputs and outputs[0]["shape"] and isinstance(outputs[0]["shape"][-1], int):
        classes = outputs[0]["shape"][-1]
    return {"inputs": inputs, "outputs": outputs, "classes": classes, "problems": problems}


def run_onnx(onnx_path: Path, input_ids, attention_mask, verbose: bool = True) -> tuple:
    """逐条跑 ONNX（批大小 1），返回 (logits, 每条耗时毫秒)。"""
    import onnxruntime as ort                              # 真实模式才需要

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    info = inspect_model(sess, expect_len=int(input_ids.shape[1]))
    if verbose:
        print("[1/3] model contract")
        for i in info["inputs"]:
            print(f"      input  {i['name']:<16} {i['shape']}  {i['type']}")
        for o in info["outputs"]:
            print(f"      output {o['name']:<16} {o['shape']}  {o['type']}")
        if info["classes"] is not None:
            print(f"      classes = {info['classes']} - the output vector carries NO field names,")
            print(f"      so the class order must be documented outside the model")
    for p in info["problems"]:
        print(f"      [WARN] {p}")

    names = [i.name for i in sess.get_inputs()]
    feats = {"input_ids": input_ids.astype(np.int64),
             "attention_mask": attention_mask.astype(np.int64)}
    feeds = [{n: feats[n][i:i + 1] for n in names if n in feats} for i in range(len(input_ids))]

    logits = np.zeros((len(input_ids), info["classes"] or 1), dtype=np.float64)
    t0 = time.time()
    for i, feed in enumerate(feeds):
        logits[i] = sess.run(None, feed)[0][0]
    ms = (time.time() - t0) / max(len(input_ids), 1) * 1000
    if verbose:
        print(f"[2/3] inference done: {len(input_ids)} item(s), {ms:.1f} ms/item")
    return logits, ms


def load_reference(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    missing = [k for k in ("input_ids", "attention_mask", "ref_logits") if k not in data]
    if missing:
        raise KeyError(f"{path} 缺少必需的键: {missing}")
    out = {k: data[k] for k in ("input_ids", "attention_mask", "ref_logits")}
    if "labels" in data:
        out["labels"] = data["labels"]
    return out


# ---------------------------------------------------------------- 报告


def report(result: dict, tol: float, elapsed_ms: float | None, labels: Sequence[str] | None,
           synthetic: bool = False) -> int:
    agree = f"{result['agreement']}/{result['items']} ({result['agreement_rate'] * 100:.1f}%)"
    print("\n[3/3] numerical comparison")
    if synthetic:
        print("      *** SYNTHETIC DEMO - no model was loaded, these numbers prove nothing ***")
    print(f"      max |delta logits| : {result['max_abs']:.2e}   (tolerance {tol:.0e})")
    print(f"      mean |delta logits|: {result['mean_abs']:.2e}")
    print(f"      argmax agreement   : {agree}")
    if elapsed_ms is not None:
        print(f"      latency            : {elapsed_ms:.1f} ms/item (CPU, batch size 1)")
    print(f"      reading            : {describe_deviation(result['max_abs'])}")

    if result["mismatch"]:
        print(f"\n      {len(result['mismatch'])} item(s) disagree, first few:")
        for i in result["mismatch"][:5]:
            name = labels[i] if labels and i < len(labels) else "?"
            print(f"        item {i}: argmax differs (label={name})")

    ok = result["within_tol"] and result["agreement"] == result["items"]
    print("\n" + "=" * 66)
    print("PASS - exported model reproduces the reference logits" if ok
          else "FAIL - see the reading line above for which layer to check")
    if ok:
        print("      class order and tokenizer still have to be documented/shipped separately;")
        print("      this check only proves the weights + graph are the same model.")
    print("=" * 66)
    return 0 if ok else 1


# ---------------------------------------------------------------- 入口


def demo(n_items: int = 16, seq_len: int = 16, classes: int = 3, seed: int = 20260101) -> dict:
    """造一批确定性输入与参考 logits，让整条比较流程在无模型时也能跑通。"""
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(0, 1000, size=(n_items, seq_len)).astype(np.int64)
    attention_mask = np.ones_like(input_ids)
    # 一个确定性"参考实现"：把 token id 的均值投影到 classes 维
    ref = np.stack([input_ids.mean(axis=1) * (c + 1) for c in range(classes)], axis=1)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "ref_logits": ref}


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ONNX vs framework numerical consistency check")
    ap.add_argument("--onnx", type=Path, help="exported .onnx file")
    ap.add_argument("--data", type=Path, help="reference .npz (input_ids/attention_mask/ref_logits)")
    ap.add_argument("--demo", action="store_true", help="synthetic run, no model needed")
    ap.add_argument("--tol", type=float, default=1e-3, help="max allowed |delta logits| (default 1e-3)")
    ap.add_argument("--labels", default="", help="comma separated class names, in model order")
    ap.add_argument("--json", type=Path, help="write the result as json")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if np is None:
        print("missing dependency: numpy  ->  python -m pip install -r requirements.txt")
        return 2

    labels = [x for x in (args.labels.split(",") if args.labels else []) if x]
    print("=" * 66)
    print("ONNX export verification")
    print("=" * 66)

    if args.demo:
        data = demo()
        result = compare(data["ref_logits"], data["ref_logits"], tol=args.tol)
        code = report(result, args.tol, elapsed_ms=None, labels=labels, synthetic=True)
        if args.json:
            args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return code

    if not args.onnx or not args.data:
        print("usage: python -m src.verify_onnx --onnx model.onnx --data reference.npz")
        print("       python -m src.verify_onnx --demo          # no model needed")
        return 2
    for p in (args.onnx, args.data):
        if not p.exists():
            print(f"not found: {p}")
            return 2

    data = load_reference(args.data)
    if len(labels) not in (0, int(data["ref_logits"].shape[1])):
        print(f"[WARN] --labels 给了 {len(labels)} 个名字，模型输出 {data['ref_logits'].shape[1]} 类")
    got, ms = run_onnx(args.onnx, data["input_ids"], data["attention_mask"])
    result = compare(data["ref_logits"], got, tol=args.tol)
    result["onnx"] = str(args.onnx)
    result["data"] = str(args.data)
    result["labels"] = labels
    code = report(result, args.tol, elapsed_ms=ms, labels=labels)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"result written to {args.json}")
    return code


if __name__ == "__main__":
    sys.exit(main())
