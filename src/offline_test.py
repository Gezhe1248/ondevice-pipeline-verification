# -*- coding: utf-8 -*-
"""进程级网络隔离测试：证明整条推理链路在断网时仍然能用。

【怎么模拟断网】
    在**当前进程**里把网络出口掐掉：
      · ``socket.getaddrinfo``      —— 除回环地址外一律解析失败（断 DNS）
      · ``socket.create_connection`` —— 除回环地址外一律抛 OSError（断 TCP）
    并把环境里的凭证变量（``*KEY*`` / ``*TOKEN*`` / ``*SECRET*``）删掉，只报变量名、
    不打印值。云端客户端一旦被创建就会立刻报错，所以任何一次意外的外呼都会**当场炸出来**，
    而不是悄悄成功。

【它同时验证的三件事（缺一条都不能算通过）】
    1. 固定轮次的多轮交互全部完成，回复非空；
    2. 全程对外连接尝试 **0 次**（负向断言）；
    3. **正向对照**：隔离确实生效 —— 隔离期间主动打一个不可路由地址必须被拦下。
       没有第 3 条，"0 次外呼"可能只说明压根没装上线，什么也证明不了。
    另外（安全架构约束）：受保护轮次**不允许**调用生成器 —— 用调用计数器证明，不靠看输出；
    回环地址必须仍然可通（本机的推理服务正是走它）。

【边界，如实说明】
    被屏蔽的是**本进程**的 socket 调用。如果推理服务是本机的另一个进程，本测试管不到它；
    那种情况应由"该进程只监听 127.0.0.1、只读本地权重文件"来保证（见 docs/engineering-note.md）。
    要做物理断网复测，拔网线/关 Wi-Fi 跑同一条命令即可。

【怎么套到你自己的流水线上】
    把 ``ScriptedPipeline`` 换成你自己的会话对象（只需要一个 ``reply(utterance) -> (text, source)``
    的方法），其余断言不用改。

用法：
    python -m src.offline_test                 # 默认跑 3 遍固定脚本（= 36 轮）
    python -m src.offline_test --rounds 1      # 快跑（= 12 轮）
"""
from __future__ import annotations

import argparse
import contextlib
import os
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .guard import GuardPolicy, enforce, strip_untrusted_numbers

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0", ""})
# RFC 5737 的测试网段：保证不可路由，用来做"隔离是否真的生效"的正向对照
UNROUTABLE_PROBE = ("192.0.2.1", 80)

CREDENTIAL_PATTERNS = (re.compile(r"KEY", re.I), re.compile(r"TOKEN", re.I),
                       re.compile(r"SECRET", re.I), re.compile(r"PASSWORD", re.I))
# 号码形状：≥5 位数字，中间允许 - — － 或空格
_PHONE_LIKE_RE = re.compile(r"\d[\d\-—－\s]{3,}\d")

# 受信号码用 RFC 预留的虚构号段，仓库里不放任何真实号码
TRUSTED_NUMBER = "555-0100"


# ---------------------------------------------------------------- 断网

def _is_loopback(host: object) -> bool:
    if not isinstance(host, str):
        return False
    return host in LOOPBACK_HOSTS or host.startswith("127.")


@contextlib.contextmanager
def blocked_network(attempts: list[tuple[str, str]]):
    """把除回环以外的网络全部掐掉，模拟拔网线。退出时一定恢复原函数。

    被拦下的每一次尝试都记进 ``attempts``：这个列表空着，才叫"对外访问 0 次"。
    """
    real_getaddrinfo = socket.getaddrinfo
    real_create_connection = socket.create_connection

    def guarded_getaddrinfo(host, *a, **kw):
        if _is_loopback(host):
            return real_getaddrinfo(host, *a, **kw)
        attempts.append(("dns", str(host)))
        raise socket.gaierror(-2, f"[offline] name resolution blocked: {host}")

    def guarded_create_connection(address, *a, **kw):
        host = address[0] if isinstance(address, tuple) else address
        if _is_loopback(host):
            return real_create_connection(address, *a, **kw)
        attempts.append(("tcp", str(host)))
        raise OSError(101, f"[offline] outbound connection blocked: {host}")

    socket.getaddrinfo = guarded_getaddrinfo
    socket.create_connection = guarded_create_connection
    try:
        yield attempts
    finally:
        socket.getaddrinfo = real_getaddrinfo
        socket.create_connection = real_create_connection


def strip_credentials() -> list[str]:
    """删掉环境里的凭证变量，返回**被删掉的变量名**（绝不打印值）。"""
    removed = []
    for name in list(os.environ):
        if any(p.search(name) for p in CREDENTIAL_PATTERNS):
            os.environ.pop(name, None)
            removed.append(name)
    return removed


def loopback_still_works(timeout: float = 2.0) -> bool:
    """隔离期间回环必须仍然可通 —— 本机推理服务走的就是这条路径。"""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    accepted = threading.Event()

    def _accept():
        try:
            conn, _ = srv.accept()
            conn.close()
        except OSError:
            pass
        finally:
            accepted.set()

    t = threading.Thread(target=_accept, daemon=True)
    t.start()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            pass
        return accepted.wait(timeout)
    except OSError:
        return False
    finally:
        srv.close()


def guard_is_active(probe_attempts: list[tuple[str, str]]) -> bool:
    """正向对照：隔离期间外呼**必须**失败。失败才说明隔离真的装上了。"""
    try:
        socket.getaddrinfo("example.invalid", 443)
    except (socket.gaierror, OSError):
        pass
    else:
        return False
    try:
        socket.create_connection(UNROUTABLE_PROBE, timeout=1)
    except OSError:
        pass
    else:
        return False
    return bool(probe_attempts)


# ---------------------------------------------------------------- 被测流水线


# 固定脚本：只分三个抽象类别，不含业务语义。
# 类别顺序即路由顺序：A/B 走生成器，C 是受保护路径（走代码里的写死文本）。
SCRIPT: tuple[tuple[str, str], ...] = (
    ("class-A", "今天在地铁上站了四十分钟，腿有点酸。"),
    ("class-A", "会议被推迟到下周三，我得重新排一遍计划。"),
    ("class-A", "房间里堆了三个没拆的箱子，一直懒得动。"),
    ("class-B", "这个月第三次改需求了，说实话有点泄气。"),
    ("class-B", "我发出去的消息没人回，有点在意这件事。"),
    ("class-B", "最近总在半夜醒，白天也没什么精神。"),
    ("class-C", "这一条需要走受保护路径。"),
    ("class-C", "这一条也需要走受保护路径。"),
    ("class-C", "这一条同样需要走受保护路径。"),
)

VERIFIED_TEXTS = {
    "class-C": f"我先停在这里。请现在联系：{TRUSTED_NUMBER}。你愿意先做一件小事吗？",
}
FALLBACK_TEXT = "我在，刚才那句我没组织好。你想先从哪里说起？"


@dataclass
class ScriptedPipeline:
    """三阶段流水线的最小实现：判定 → 路由 → 后处理。

    ⚠️ 这里的"判定"是**占位实现**：类别由脚本直接给出，好让这份测试在没有模型的情况下
       也能跑通。生产里把它换成真正的分类器即可 —— 被验证的是隔离与路由，不是判定准确率。
    """

    generator: Callable[[str], str]
    verified: dict[str, str] = field(default_factory=lambda: dict(VERIFIED_TEXTS))
    guard_policy: GuardPolicy = field(
        default_factory=lambda: GuardPolicy(trusted_numbers=frozenset({TRUSTED_NUMBER})))
    generator_calls: int = 0

    def reply(self, tier: str, utterance: str) -> tuple[str, str]:
        """返回 (回复文本, 来源)。来源：``verified`` / ``generator``。"""
        if tier in self.verified:               # 受保护路径：绝不调用生成器
            return self.verified[tier], "verified"
        self.generator_calls += 1
        raw = self.generator(utterance)
        raw, _ = strip_untrusted_numbers(raw)   # 号码只能来自写死文本
        text, _ = enforce(raw, self.guard_policy)
        return (text or FALLBACK_TEXT), "generator"


def stub_generator(utterance: str) -> str:
    """占位生成器：确定性、无网络、无模型。

    故意在第 4 轮吐出一个号码形状，用来验证"生成通道里的号码一律被删"这条约束
    是真的在生效（故障注入，不是装饰）。
    """
    if "第三次改需求" in utterance:
        return "听起来确实泄气。你可以打 400-123-4567 找人聊聊。要不要说说具体卡在哪？"
    head = utterance.rstrip("。！？!?，,")
    return f"关于「{head}」，我想先听你说完。你最在意的是其中的哪一部分？"


# ---------------------------------------------------------------- 主流程


def run(rounds: int = 3, verbose: bool = True) -> int:
    removed = strip_credentials()
    print("=" * 74)
    print("offline test: every non-loopback socket call is blocked in this process")
    print("=" * 74)
    print(f"  credentials removed : {', '.join(removed) if removed else '(none found)'}")
    print(f"  blocked surfaces    : socket.getaddrinfo / socket.create_connection")
    print(f"  loopback allowed    : {', '.join(sorted(h or '<empty>' for h in LOOPBACK_HOSTS))}")

    plan: Sequence[tuple[str, str]] = [round_ for _ in range(max(rounds, 1)) for round_ in SCRIPT]
    pipeline = ScriptedPipeline(generator=stub_generator)
    attempts: list[tuple[str, str]] = []
    results: list[dict] = []

    t0 = time.time()
    with blocked_network(attempts):
        loop_ok = loopback_still_works()
        for i, (tier, utterance) in enumerate(plan, 1):
            before = pipeline.generator_calls
            start = time.time()
            text, source = pipeline.reply(tier, utterance)
            used_generator = pipeline.generator_calls > before
            results.append({"round": i, "tier": tier, "utterance": utterance,
                            "reply": text, "source": source,
                            "used_generator": used_generator,
                            "seconds": round(time.time() - start, 3)})
            if verbose:
                print(f"\n[{i:02d}] {tier} source={source} "
                      f"generator_calls+={int(used_generator)} "
                      f"{results[-1]['seconds']}s")
                print(f"     user : {utterance}")
                print(f"     reply: {text}")
    elapsed = time.time() - t0

    # 正向对照放在隔离区之外重新进一次，避免把它自己的尝试混进上面的计数器
    control_attempts: list[tuple[str, str]] = []
    with blocked_network(control_attempts):
        control_ok = guard_is_active(control_attempts)

    # ---- 断言 ----
    checks: list[tuple[str, bool, str]] = []

    nonempty = all(r["reply"].strip() for r in results)
    checks.append((f"all {len(results)} round(s) produced a non-empty reply",
                   nonempty, f"{sum(1 for r in results if r['reply'].strip())}/{len(results)}"))

    checks.append(("outbound network attempts during the session",
                   not attempts, "0" if not attempts else str(attempts[:3])))

    checks.append(("positive control: the block really rejects outbound traffic",
                   control_ok, "blocked as expected" if control_ok else "NOT blocked!")) 

    checks.append(("loopback still reachable under the block",
                   loop_ok, "ok" if loop_ok else "loopback broken"))

    protected_rounds = [r for r in results if r["tier"] in VERIFIED_TEXTS]
    protected_clean = all((not r["used_generator"]) for r in protected_rounds)
    checks.append((f"protected rounds never call the generator ({len(protected_rounds)} rounds)",
                   protected_clean,
                   f"{sum(1 for r in protected_rounds if r['used_generator'])} violation(s)"))

    generator_calls_expected = len(results) - len(protected_rounds)
    calls_ok = pipeline.generator_calls == generator_calls_expected
    checks.append((f"generator called exactly {generator_calls_expected} time(s)",
                   calls_ok, f"{pipeline.generator_calls}"))

    leaked = [r["reply"] for r in results
              if r["source"] == "generator" and _PHONE_LIKE_RE.search(r["reply"])]
    checks.append(("no number survived the generator path", not leaked,
                   "0" if not leaked else str(leaked[:1])))

    protected_number_ok = all(TRUSTED_NUMBER in r["reply"] for r in protected_rounds)
    checks.append(("protected replies carry the verified number unchanged",
                   protected_number_ok, TRUSTED_NUMBER))

    print("\n" + "-" * 74)
    failures = 0
    for name, ok, detail in checks:
        failures += 0 if ok else 1
        print(f"[{'OK  ' if ok else 'FAIL'}] {name}  ->  {detail}")
    print("-" * 74)
    print(f"rounds {len(results)}  |  outbound attempts {len(attempts)}  |  "
          f"elapsed {elapsed:.2f}s  |  "
          f"{'PASS' if not failures else f'{failures} CHECK(S) FAILED'}")
    return 0 if failures == 0 else 1


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="process-level network isolation test")
    ap.add_argument("--rounds", type=int, default=3,
                    help="repeat the fixed script N times (default 3 -> %d rounds)" % (3 * len(SCRIPT)))
    ap.add_argument("--quiet", action="store_true", help="only print the check list")
    args = ap.parse_args(list(argv) if argv is not None else None)
    return run(rounds=args.rounds, verbose=not args.quiet)


if __name__ == "__main__":
    sys.exit(main())
