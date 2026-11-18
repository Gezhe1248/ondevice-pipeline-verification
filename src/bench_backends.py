# -*- coding: utf-8 -*-
"""推理后端对照实验：同一个模型换后端，测启动成本与生成速度。

【为什么值得单独测一次】
    "本地模型启动慢"很容易被当成模型固有的成本写进结论，然后就没有人再问"能不能快一点"。
    实测下来启动慢的往往不是模型，而是**后端的初始化方式**：有的后端在首次请求时
    运行时编译 kernel/着色器（十几秒），有的后端预编译（不到一秒）。这不是优化问题，
    是选型问题——而且用户每次启动都要等。

【测什么，以及哪些不该当结论】
    1. 启动：模型加载 + 首次请求，分开记。合成一个数就看不出慢在哪一步。
    2. 生成速度：多个提示词 × 多次重复，报**中位数与区间**。单样本会骗人。
    3. 质量：后端不该改变输出质量，所以同一批提示词的复读率要一起报，
       否则"快但变傻"看不出来（复读检测复用 guard 的句级查重）。
    ⚠️ 生成速度的差别常常在噪声内：同一台机器两轮测量，两个后端的名次可以**正好对调**。
       所以结论只能落在"结构性差异"（启动路径）上，速度差异要写成"无可辨别差异"。

【两种跑法】
    · 真实后端（需要 llama.cpp 之类的本地 HTTP 服务）：
          python -m src.bench_backends --name cuda --api-base http://127.0.0.1:8080 \
                 --launch-cmd "path/to/server -m model.gguf -ngl 99" --repeat 2
      不给 --launch-cmd 就只测"已经在跑的服务"，启动耗时记为 not measured。
    · --stub：**合成数据**，不启动任何服务，只用来演示测量与报告流程。
      输出会打上 SYNTHETIC 标记——这些数字不能引用。

用法：
    python -m src.bench_backends --stub
    python -m src.bench_backends --name cuda --api-base http://127.0.0.1:8080 --repeat 2
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .guard import drop_repeated_sentences

# 覆盖不同长度的固定提示词（长度会影响生成速度，只看一种会得出片面结论）
PROMPTS: tuple[str, ...] = (
    "今天在地铁上站了四十分钟，腿有点酸。",
    "会议被推迟到下周三，我得重新排一遍计划。",
    "房间里堆了三个没拆的箱子，一直懒得动。",
    "这个月第三次改需求了，说实话有点泄气。",
    "我发出去的消息没人回，有点在意这件事。",
    "最近总在半夜醒，白天也没什么精神。",
)


@dataclass
class BackendResult:
    name: str
    load_seconds: float | None = None       # 模型加载（进程起到服务 ready）
    first_request_seconds: float | None = None
    startup_seconds: float | None = None    # 上面两项之和 = 用户每次启动要等的
    gen_speed_median: float | None = None   # tok/s
    gen_speed_range: list | None = None
    samples: int = 0
    repeat_rate: float | None = None        # 句级复读率（%），后端不该改变它
    replies: list = field(default_factory=list)
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------- 生命周期


def _wait_health(api_base: str, timeout: float) -> float | None:
    """轮询 /health 直到就绪，返回等待秒数。"""
    deadline = time.time() + timeout
    url = api_base.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return None
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.1)
    return None


class Backend:
    """一个后端 = 一个 API 地址（可选：自己拉起进程）。"""

    def __init__(self, name: str, api_base: str, launch_cmd: str | None = None,
                 launch_timeout: float = 180.0):
        self.name, self.api_base = name, api_base
        self.launch_cmd, self.launch_timeout = launch_cmd, launch_timeout
        self.proc: subprocess.Popen | None = None

    def start(self) -> float | None:
        """拉起进程（若有）并等到 ready。返回模型加载秒数；未启动则 None。"""
        if not self.launch_cmd:
            return None
        t0 = time.time()
        self.proc = subprocess.Popen(shlex.split(self.launch_cmd, posix=(os.name != "nt")),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + self.launch_timeout
        while time.time() < deadline:
            if _server_ready(self.api_base):
                return time.time() - t0
            if self.proc.poll() is not None:
                raise RuntimeError(f"backend exited during startup: {self.name}")
            time.sleep(0.1)
        raise TimeoutError(f"backend not ready within {self.launch_timeout}s: {self.name}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def complete(self, prompt: str, n_predict: int, timeout: float = 120.0) -> dict:
        """调一次补全接口，返回原始响应（含 timings）。"""
        payload = json.dumps({"prompt": prompt, "n_predict": n_predict,
                              "temperature": 0.0}).encode("utf-8")
        req = urllib.request.Request(self.api_base.rstrip("/") + "/completion",
                                     data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"), strict=False)


def _server_ready(api_base: str) -> bool:
    host, _, port = api_base.split("//")[-1].partition(":")
    try:
        with socket.create_connection((host or "127.0.0.1", int(port or 80)), timeout=1):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------- 测量


def repeat_rate(replies: Sequence[str], threshold: float = 0.80) -> float:
    """句级复读率：有多少条回复里出现了前面某轮已经说过的句子（%）。"""
    if len(replies) < 2:
        return 0.0
    hit = 0
    for i, reply in enumerate(replies):
        if i == 0:
            continue
        kept, _ = drop_repeated_sentences(reply, replies[:i], threshold=threshold)
        if kept.strip() != reply.strip():
            hit += 1
    return round(hit / (len(replies) - 1) * 100, 1)


def measure(backend: Backend, prompts: Sequence[str], repeat: int, n_predict: int,
            verbose: bool = True) -> BackendResult:
    res = BackendResult(name=backend.name)
    res.load_seconds = backend.start()
    if res.load_seconds is None:
        res.notes.append("launch command not given: startup cost not measured")
        _wait_health(backend.api_base, timeout=10)

    # 首次请求单独计时：有的后端把一次性编译成本放在这一步
    t0 = time.time()
    backend.complete(prompts[0], n_predict=4)
    res.first_request_seconds = round(time.time() - t0, 3)
    if res.load_seconds is not None:
        res.startup_seconds = round(res.load_seconds + res.first_request_seconds, 1)

    # 预热到稳态，再量速度（否则量到的是"第一次"而不是"每次"）
    for p in prompts:
        backend.complete(p, n_predict=64)

    speeds, replies = [], []
    for _ in range(max(repeat, 1)):
        for p in prompts:
            t0 = time.time()
            data = backend.complete(p, n_predict=n_predict)
            elapsed = time.time() - t0
            timings = data.get("timings") or {}
            speed = timings.get("predicted_per_second")
            if not speed:
                tokens = (data.get("usage") or {}).get("completion_tokens") or n_predict
                speed = tokens / elapsed if elapsed > 0 else 0.0
            speeds.append(float(speed))
            replies.append((data.get("content") or "").strip())

    res.samples = len(speeds)
    res.gen_speed_median = round(statistics.median(speeds), 1)
    res.gen_speed_range = [round(min(speeds), 1), round(max(speeds), 1)]
    res.replies = replies
    res.repeat_rate = repeat_rate(replies)
    if verbose:
        print(f"  {backend.name}: startup={res.startup_seconds}s "
              f"gen={res.gen_speed_median} tok/s {res.gen_speed_range} "
              f"repeat={res.repeat_rate}%")
    return res


# ---------------------------------------------------------------- stub


def measure_stub(prompts: Sequence[str], repeat: int, n_predict: int) -> list[BackendResult]:
    """合成两行结果，只用来演示报告格式。**这些数字不能引用。**"""
    rows = []
    for name, load, first, speed in (("stub-A", 0.8, 0.2, 120.0), ("stub-B", 0.5, 3.0, 118.0)):
        replies = [f"（合成回复 {i}）针对「{p[:8]}」的一句回应。" for i, p in enumerate(prompts)]
        rows.append(BackendResult(
            name=name, load_seconds=load, first_request_seconds=first,
            startup_seconds=round(load + first, 1),
            gen_speed_median=speed, gen_speed_range=[speed - 3, speed + 3],
            samples=len(prompts) * max(repeat, 1), repeat_rate=0.0, replies=replies,
            notes=["synthetic: produced by --stub, not a real measurement"]))
    return rows


# ---------------------------------------------------------------- 报告


def report(rows: Sequence[BackendResult], synthetic: bool, md_path: Path | None = None) -> None:
    def fmt(v, unit=""):
        return "not measured" if v is None else f"{v}{unit}"

    head = ["metric", *[r.name for r in rows]]
    table = [
        ("startup total (s)", [fmt(r.startup_seconds) for r in rows],
         "what the user waits for on every launch"),
        ("  model load (s)", [fmt(r.load_seconds) for r in rows], ""),
        ("  first request (s)", [fmt(r.first_request_seconds) for r in rows],
         "one-off kernel/shader compilation lands here"),
        ("gen speed median (tok/s)", [fmt(r.gen_speed_median) for r in rows], "higher is better"),
        ("gen speed range (tok/s)", [fmt(r.gen_speed_range) for r in rows], ""),
        ("samples (n)", [fmt(r.samples) for r in rows], ""),
        ("repeat rate (%)", [fmt(r.repeat_rate) for r in rows],
         "quality: a backend must not change this"),
    ]

    print("\n" + "=" * 74)
    print("backend comparison" + ("   *** SYNTHETIC (--stub): DO NOT QUOTE ***" if synthetic else ""))
    print("=" * 74)
    widths = [max(len(str(c)) for c in col) for col in
              ([h for h, _, _ in table] + [head])]
    for name, values, note in table:
        cells = [name, *values]
        print("  " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))
              + (f"   # {note}" if note else ""))
    notes = list(dict.fromkeys(n for r in rows for n in r.notes))
    if notes:
        print("\n  notes:")
        for n in notes:
            print(f"    - {n}")
    print("\n  reading guide:")
    print("    - startup dominates the user experience; generation speed differences between")
    print("      backends are usually inside run-to-run noise on the same machine - report")
    print("      'no discernible difference' unless the gap survives repeated runs.")
    print("    - repeat rate is a quality guard: if it moves, the backend is producing")
    print("      different text, which invalidates the speed comparison.")
    print("=" * 74)

    if md_path:
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for name, values, note in table:
            lines.append("| " + " | ".join([name, *values]) + " |")
        if synthetic:
            lines.insert(0, "> SYNTHETIC --stub output, do not quote.\n")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"markdown written to {md_path}")


# ---------------------------------------------------------------- 入口


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="cold start / throughput comparison across backends")
    ap.add_argument("--name", default="backend", help="label printed in the report")
    ap.add_argument("--api-base", default="http://127.0.0.1:8080", help="base URL of the server")
    ap.add_argument("--launch-cmd", default=None,
                    help='command that starts the server, e.g. "server.exe -m model.gguf"')
    ap.add_argument("--repeat", type=int, default=2, help="repeat the prompt set N times (default 2)")
    ap.add_argument("--n-predict", type=int, default=160, help="tokens per request (default 160)")
    ap.add_argument("--stub", action="store_true", help="synthetic demo, no backend needed")
    ap.add_argument("--json", type=Path, help="write raw results as json")
    ap.add_argument("--md", type=Path, help="write the table as markdown")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.stub:
        rows = measure_stub(PROMPTS, args.repeat, args.n_predict)
    else:
        backend = Backend(args.name, args.api_base, args.launch_cmd)
        try:
            rows = [measure(backend, PROMPTS, args.repeat, args.n_predict)]
        except (urllib.error.URLError, OSError, TimeoutError, RuntimeError) as e:
            print(f"[x] {type(e).__name__}: {e}")
            print("    start the server first, or pass --launch-cmd, or use --stub for a demo")
            return 1
        finally:
            backend.stop()

    report(rows, synthetic=args.stub, md_path=args.md)
    if args.json:
        args.json.write_text(json.dumps([r.__dict__ for r in rows], ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"raw results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
