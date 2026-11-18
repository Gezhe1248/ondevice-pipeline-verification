# -*- coding: utf-8 -*-
"""确定性后处理层（deterministic post-processing layer for generated replies）。

【定位】
    生成器的输出在交给用户之前先经过这一层。它只做两件事：**删句子、去符号**。
    不合成新句、不改写语义、不新增内容。

【三条不变量（invariants）】
    1. **受保护内容原样返回**：调用方显式传 ``is_protected=True``，或文本里出现
       受信号码（``policy.trusted_numbers``）时，一个字都不改。
       裁掉安全信息属于事故级错误，不是风格问题。
    2. **只删不编**：任何规则都只能减少内容，不能补内容。
    3. **绝不返回空**：即使所有句子都被判为不合格，也至少保留原句。
       宁可啰嗦，不可失语。

【它不做什么】
    不做分类、不做生成、不判断内容是否安全——那是上游判定器与写死文本模块的事。
    正确的调用顺序：上游判定 → 命中受保护路径走写死文本 → 未命中才走生成器 + 本模块。

【为什么需要它：提示词只能提高概率，不能保证】
    - 同一批固定输入上，把"必须在特定条件下才生效"的规则写进提示词，实测该行为
      9 → 8 条（落在噪声内，等于无效）；把同一个条件判断搬进代码，8 → 4 条。
      小模型执行不了带条件的规则，只能由代码在**知道上下文**的前提下做确定性处理。
    - 所以凡"必须成立"的约束都落在这一层，且每条规则都配可断言的用例（见 tests/）。
    - 反面教材（已修）：曾用一份 17 词的词表做**整段短路**——回复里出现其中任一个词，
      整个模块原样放行，于是句数上限、问号上限、去符号、删说教句四条规则同时失效；
      而那份词表里的"计划""安全"这类普通词在日常文本里太常见，等于随机失效。
      现在这类豁免一律做到**句级**：只保护真正需要保护的句子，其余照常修剪。

用法：
    from guard import GuardPolicy, enforce, strip_untrusted_numbers

    policy = GuardPolicy(trusted_numbers=frozenset({"555-0100"}))  # 你自己核对过的号码
    text, info = enforce(raw_text, policy)
    text, info = strip_untrusted_numbers(raw_text)                 # 生成通道专用
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

__all__ = [
    "GuardPolicy",
    "enforce",
    "drop_repeated_sentences",
    "drop_redundant_questions",
    "has_trusted_number",
    "question_count",
    "split_sentences",
    "strip_foreign_words",
    "strip_marks",
    "strip_repeated_opener",
    "strip_report_frame",
    "strip_untrusted_numbers",
    "untrusted_numbers",
]

# ---------------------------------------------------------------- 策略

# 「在说教」的标志词：一出现就基本等于在教育对方、否定其自主性。
#
# ⚠️ 判据必须**窄**。反面教材：曾把「方法」「办法」这两个普通名词也算成"在给建议"，
#    于是「听到你找到了问题的**解决办法**」这种描述对方自己行为的正常句被整句删掉，
#    用户看到的就是答非所问。同理，具体行为词（"去散步""深呼吸"）不是说教句式——
#    把它们一起删，等于把有用的建议整段阉割。判据宽 → 误删；误删比漏删更难被发现。
PRESCRIPTIVE_MARKERS: tuple[str, ...] = (
    "建议你", "我建议", "你应该", "你不应该", "你必须", "教你",
    "第一步", "第二步", "这样做", "几个方法",
)

# 纯填充型开场白：不含任何关于对方的信息，单独出现等于没在听。
# ⚠️ 只列**确定是空话**的；"我理解你现在的处境"这类带内容的不要放进来。
FILLER_OPENERS: tuple[str, ...] = (
    "你说得对", "你说的对", "我理解你", "我听到了", "我明白", "我懂你",
    "我在这里", "嗯，我理解",
)

# 「汇报腔」开头：把对方刚说过的话原样复述一遍再回答。
#
# ⚠️ 两条铁律（实测踩到后定下）：
#   1. 匹配顺序 = 从长到短，否则「你说的」先命中，会把「你说的那些」拆成残句；
#   2. **只列"提到 / 刚才说 / 说的那些"这类无歧义的汇报词**。
#      坚决不列光板「你说」「你说的」——它们可能是正常句子的开头：
#        「你说话的时候情绪很重」→ 剥掉「你说」就变成残句「话的时候情绪很重」。
#      宁可漏剥一个温和的汇报腔，也不能把正常句子削断。
REPORT_FRAMES: tuple[str, ...] = (
    "你刚才说的那些", "你刚才说的这些", "你刚才提到", "你刚才说的", "你刚才说",
    "你说的那些", "你说的这些", "你说过的", "你提到的", "你提到",
)
REPORT_KEEP: tuple[str, ...] = ("你说得对", "你说的对")   # 这两句交给 FILLER_OPENERS

# 句末标点切句（保留标点，逐句删除时要能原样拼回去）
_SENT_SPLIT = re.compile(r"(?<=[。！？!?…])")
_BULLET_RE = re.compile(r"^\s*[-*•·]\s+")
_NORM_STRIP_RE = re.compile(r"[\s，。！？、；：,.!?;:—\-－~…\"'“”（）()【】\[\]]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
# 号码形状：≥5 位数字，中间允许出现 - — － 或空格
_PHONE_LIKE_RE = re.compile(r"\d[\d\-—－\s]{3,}\d")
QUESTION_MARKS: tuple[str, ...] = ("？", "?")


@dataclass(frozen=True)
class GuardPolicy:
    """后处理策略。所有阈值都是**显式参数**，不在代码里散落魔法数字。

    Attributes:
        max_sentences: 句数上限（上游约定写 2-4 句时取 4；兜底不该比上游约定更严）。
        max_questions: 全篇问号上限，``None`` = 不限制。
        drop_prescriptive: 是否删掉说教句式（见 ``PRESCRIPTIVE_MARKERS``）。
        trusted_numbers: 受信号码（人工核对过的）。命中即整段原样返回。
        protected_predicate: 句级豁免谓词——返回 True 的句子不参与问号上限、
            也不会在截断时被丢掉。``None`` = 不豁免任何句子。
    """

    max_sentences: int = 4
    max_questions: int | None = 1
    drop_prescriptive: bool = True
    trusted_numbers: frozenset[str] = field(default_factory=frozenset)
    protected_predicate: Callable[[str], bool] | None = None


# ---------------------------------------------------------------- 基础工具


def split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句，保留标点。"""
    return [p.strip() for p in _SENT_SPLIT.split(text or "") if p and p.strip()]


def question_count(text: str) -> int:
    """全篇问号数（半角全角都算）。"""
    return sum((text or "").count(c) for c in QUESTION_MARKS)


def strip_marks(text: str) -> str:
    """去掉 markdown / 分点符号，保留文字。分点符号念出来像说明书。"""
    out = []
    for line in (text or "").splitlines():
        line = _BULLET_RE.sub("", line.strip())
        line = line.replace("**", "").replace("###", "").replace("##", "")
        out.append(line.strip())
    return "\n".join(x for x in out if x)


def untrusted_numbers(text: str) -> list[str]:
    """文本里出现的号码形状片段（用于报告"拦下了什么"）。"""
    return _PHONE_LIKE_RE.findall(text or "")


def has_trusted_number(text: str, policy: GuardPolicy | None = None) -> bool:
    """是否含受信号码 —— 含则本模块完全不介入。"""
    policy = policy or GuardPolicy()
    if not policy.trusted_numbers:
        return False
    flat = re.sub(r"[\s\-—－]", "", text or "")
    return any(re.sub(r"[\s\-—－]", "", n) in flat for n in policy.trusted_numbers)


def _norm_plain(text: str) -> str:
    """去掉标点空白，只留字面，用于逐句比相似度。"""
    return _NORM_STRIP_RE.sub("", text or "")


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm_plain(a), _norm_plain(b)).ratio()


def _is_protected(sentence: str, policy: GuardPolicy) -> bool:
    return bool(policy.protected_predicate and policy.protected_predicate(sentence))


# ---------------------------------------------------------------- 逐句规则


def drop_repeated_sentences(text: str, prev_replies: Sequence[str],
                            threshold: float = 0.90) -> tuple[str, dict]:
    """删掉与最近几轮助手回复**重复的句子**，返回 (文本, 说明)。

    【为什么必须做到句级，而不是整段查重】
        模型会**换个说法重复同一件事**（"花的心思" → "付出的心思"）：整段相似度只有
        0.5 左右，整段查重会放行；但用户看到的就是"又在说同一件事"。
        句级比对能把重复掉的那句删掉，只保留这一轮真正新增的内容。只删不编。
        删空时返回空串，由调用方决定要不要退到固定兜底话术。
    """
    raw = (text or "").strip()
    if not raw or not prev_replies:
        return raw, {"actions": []}
    prev_sents = [s for p in prev_replies for s in split_sentences(p)]
    if not prev_sents:
        return raw, {"actions": []}
    kept, dropped = [], 0
    for s in split_sentences(raw):
        if max((_sim(s, ps) for ps in prev_sents), default=0.0) >= threshold:
            dropped += 1
            continue
        kept.append(s)
    out = "".join(kept).strip()
    actions = ([] if not dropped else
               [f"dropped {dropped} sentence(s) already said in a previous turn "
                f"(sentence-level similarity >= {threshold})"])
    return out, {"actions": actions}


def strip_repeated_opener(text: str, prev_replies: Sequence[str] | None = None,
                          max_len: int = 6) -> tuple[str, dict]:
    """删掉开头的**空话**或**与最近几轮重复的短开场白**。

    【为什么要在输出端做】
        实测：提示词里明写"不要用同一句开场白反复开头"也压不住——18 轮里有 9 轮用
        同一句开场白，而且写明这条之后**反而更多了**。原因是自我强化：模型从自己
        上文那几条回复里学会了这个开场白，下一轮又照着学。只能靠确定性规则兜。

    【两种触发条件】
        1. 开头那一小句是 ``FILLER_OPENERS`` 里的纯填充语 —— 删；
        2. 开头那一小句 ≤ ``max_len`` 字，且与最近某轮的开头完全一致 —— 删。
        ⚠️ ``max_len`` 必须小（默认 6）：空话开场白都是 4~6 字，带内容的开头更长。
           阈值放到 10 会把"带内容回引"的开头一起删掉，等于把刚修好的回引又削回去。
        删完没内容则原样返回（绝不返回空）。
    """
    raw = (text or "").strip()
    if not raw:
        return raw, {"actions": []}
    m = re.match(r"^(.*?[，。！？!?])", raw)
    if not m:
        return raw, {"actions": []}
    core = m.group(1).rstrip("，。！？!? ").strip()
    if not core or len(core) > max_len:
        return raw, {"actions": []}

    reason = None
    if core in FILLER_OPENERS:
        reason = f"dropped filler opener: {core!r}"
    else:
        for p in (prev_replies or []):
            pm = re.match(r"^(.*?[，。！？!?])", (p or "").strip())
            if pm and pm.group(1).rstrip("，。！？!? ").strip() == core:
                reason = f"dropped opener repeated from a previous turn: {core!r}"
                break
    if not reason:
        return raw, {"actions": []}
    rest = raw[m.end():].lstrip("，。！？!? ").strip()
    if not rest:
        return raw, {"actions": []}          # 删了就没内容了 → 不删
    return rest, {"actions": [reason]}


def strip_report_frame(text: str) -> tuple[str, dict]:
    """去掉开头的「汇报腔」，**保留后面的内容**。返回 (文本, 说明)。

    【为什么只剥框架词、不整句删】
        模型每轮都用"你说的 X… / 你提到的 X…"把对方的话复述一遍，而且还会给复述**加戏**
        （对方说的是 A，复述成"你喜欢 A"）。但整句删会把内容一起删掉（"你提到的橘子"
        整句删掉就没了"橘子"），所以这里只剥掉框架词，后面的话原样留下：
            "你提到的橘子，是不是一直记得呢？" → "橘子，是不是一直记得呢？"
        剩余太短（<4 字）就原样返回，避免删成残句。只删不编，且信息不丢。
    """
    raw = (text or "").strip()
    if not raw or raw.startswith(REPORT_KEEP):
        return raw, {"actions": []}
    for fr in REPORT_FRAMES:
        if raw.startswith(fr):
            rest = raw[len(fr):].lstrip("的了着过").lstrip("，,。.：: ").strip()
            if len(rest) >= 4:
                return rest, {"actions": [f"stripped report frame: {fr!r}"]}
            return raw, {"actions": []}
    return raw, {"actions": []}


def strip_foreign_words(text: str, trusted_numbers: Iterable[str] = ()) -> tuple[str, dict]:
    """删掉含英文单词的整句，返回 (文本, 说明)。

    【为什么要有这一层】
        纯中文通道的回复里蹦出英文词，念出来很怪；而且出现英文词说明这次生成已经跑偏，
        不值得原样交给用户。只删不编；删空则由调用方走固定兜底话术。
        含受信号码时不介入（号码所在句可能是必须原样送达的文本，不能删）。

    ⚠️ 这里的豁免只认**号码**，不认"关键词"——按词豁免会导致普通词（"计划""安全"）
       把英文词一起豁免掉，规则静默失效（详见模块 docstring 的反面教材）。
    """
    policy = GuardPolicy(trusted_numbers=frozenset(trusted_numbers))
    raw = (text or "").strip()
    if not raw or not _LATIN_WORD_RE.search(raw):
        return raw, {"actions": []}
    if has_trusted_number(raw, policy):
        return raw, {"actions": ["contains a trusted number: leaving as is"]}
    kept, dropped = [], 0
    for s in split_sentences(raw):
        if _LATIN_WORD_RE.search(s):
            dropped += 1
            continue
        kept.append(s)
    out = "".join(kept).strip()
    actions = ([] if not dropped else
               [f"dropped {dropped} sentence(s) containing latin words"])
    return out, {"actions": actions}


def strip_untrusted_numbers(text: str) -> tuple[str, dict]:
    """删掉含电话号码的**整句**，返回 (文本, 说明)。**只给生成通道用。**

    【为什么必须有这一层】
        号码只能来自**人工核对过的写死文本**。生成式模型没有"号码正确"的保证。
        实测过的漏洞：上一轮的写死文本会作为 assistant 消息进入上下文，下一轮生成时
        模型把它连同号码一起模仿了出来 —— 这一次它抄对了，但没有任何机制保证它每次
        都抄对，而抄错的代价是给用户一个打不通的号码。

        所以这里做**确定性拦截**：输出里只要出现号码形状，含号码的整句一律删掉。
        只删不编；删空则返回空串，由调用方走固定兜底话术（绝不凭空造内容）。
    """
    raw = (text or "").strip()
    if not raw:
        return "", {"actions": ["empty input"]}
    if not _PHONE_LIKE_RE.search(raw):
        return raw, {"actions": []}
    kept, dropped = [], 0
    for s in split_sentences(raw):
        if _PHONE_LIKE_RE.search(s):
            dropped += 1
            continue
        kept.append(s)
    out = "".join(kept).strip()
    nums = ", ".join(untrusted_numbers(raw)[:3])
    actions = [f"dropped {dropped} sentence(s) containing a number "
               f"(numbers may only come from the verified text module; saw: {nums})"]
    return out, {"actions": actions, "dropped_numbers": untrusted_numbers(raw)}


# ---------------------------------------------------------------- 收尾轮的问句处理

# 「该收尾」时的问句处理：用户刚把事情说完（约好了时间、确认了信息），
# 回复里不该再让他确认一遍（"你准备好了吗""你确认要准时来吗"）。
#
# 但规则不是"删掉所有问句"，而是**只删冗余确认型追问**：
#   · 给选项型（"需要改时间吗"）→ 保留，这是在给对方选择权；
#   · 自然关心型（"最近感觉怎么样"）→ 保留；
#   · 冗余确认型 → 尝试切出陈述头，切不出就整句删。
# 两份表都可审可改；改动后必须跑 tests/ 里对应的用例。
_OPTION_QUESTION_RE = re.compile(
    r"(需要|要不要|是否)(更改|改一下|改|调整|换|推迟|改期)|"
    r"时间.{0,4}方便吗|方便(改|换)(个)?时间吗|要(不要)?改(个)?时间"
)
_REDUNDANT_QUESTION_RE = re.compile(
    r"准备好了?吗|准备好了?吧|你(确认|确定)[^。！？]{0,8}吗|准时来吗|你愿意等|"
    r"是不是[^。！？]{0,6}(想确认|呢)|对吗|好吗|可以吗|行吗|没问题吧"
)


def _is_redundant_question(sentence: str) -> bool:
    """这句问句是不是"催对方再确认一遍"（该收尾时应当删掉）。"""
    if not any(m in sentence for m in QUESTION_MARKS):
        return False
    if _OPTION_QUESTION_RE.search(sentence):
        return False
    return bool(_REDUNDANT_QUESTION_RE.search(sentence))


def _statement_head(sentence: str) -> str:
    """把"确认 + 催确认？"这种同句混写切成陈述句：取第一个逗号之前的部分。

    只在**问句部分已判定为冗余确认**时使用，且要求切出来的头本身是干净陈述
    （≥4 字、不含疑问词），否则返回空串 → 调用方整句删。
    依据：验收标准明确"确认要保住，只去掉追问"。
    """
    head = re.split(r"[？?]", sentence)[0].strip()
    if not head:
        return ""
    parts = [p.strip() for p in re.split(r"[，,、；;：:]", head) if p.strip()]
    if len(parts) < 2:
        return ""
    cand = parts[0]      # "周六下午三点见，你准备好了吗" → 取"周六下午三点见"
    if len(cand) < 4 or re.search(r"[吗呢么]|什么|怎么|如何|是否|对不对|好不好", cand):
        return ""
    return cand.rstrip("。！!…，,、；;：: ") + "。"


def drop_redundant_questions(text: str,
                             trusted_numbers: Iterable[str] = ()) -> tuple[str, dict]:
    """收尾轮：删掉/切掉**冗余确认型**问句（不是删所有问句）。返回 (文本, 说明)。

    【为什么放在输出端】
        先试过纯提示词条款，实测该收尾追问 9 → 8 条（噪声内）= 无效。这类带条件的规则
        小模型执行不了，只能由代码在**知道用户说了什么**的前提下做确定性处理：
        调用方只在检测到"该收尾"时才调用本函数。
        效果（配对实验，同一份原始输出套两种规则）：命中收尾 11 条 → 实改 7 条、
        **改后为空 0 条**；非收尾 105 条**零影响**。配对实验是必须的：两组各自重新
        采样得出的差异是噪声（同一配置先后跑出过 30.6%~72.2% 的"提升"）。

    处理规则（逐句）：
        · 没有问号 → 保留；
        · 给选项型问句 → 保留；
        · 冗余确认型问句 → 尝试切出陈述头，切不出就整句删；
        · 其余问句（自然关心类）→ 保留。

    删完若为空则**原样返回**；含受信号码时不动。只删不编。
    """
    raw = (text or "").strip()
    if not raw:
        return raw, {"actions": []}
    if has_trusted_number(raw, GuardPolicy(trusted_numbers=frozenset(trusted_numbers))):
        return raw, {"actions": ["contains a trusted number: questions left untouched"]}

    kept, cut, dropped = [], 0, 0
    for s in split_sentences(raw):
        if not _is_redundant_question(s):
            kept.append(s)
            continue
        head = _statement_head(s)
        if head:
            kept.append(head)
            cut += 1
        else:
            dropped += 1
    out = "".join(kept).strip()
    if not (cut or dropped):
        return raw, {"actions": []}
    if not out:
        return raw, {"actions": [f"all {cut + dropped} sentence(s) were confirmation "
                                 f"prompts -> not deleted (never return empty)"]}
    parts = []
    if cut:
        parts.append(f"{cut} kept as a plain statement")
    if dropped:
        parts.append(f"{dropped} deleted")
    return out, {"actions": ["closing turn: no more confirmation prompts (" + ", ".join(parts) + ")"]}


# ---------------------------------------------------------------- 主入口


def enforce(text: str, policy: GuardPolicy | None = None,
            is_protected: bool = False) -> tuple[str, dict]:
    """把回复压到策略允许的形状。返回 (处理后文本, 说明字典)。

    处理顺序：去符号 → 删说教句 → 只留 max_questions 个问句 → 截断到 max_sentences 句。
    任一环节都不改变其余句子的原文。

    ⚠️ ``max_sentences`` 默认 **4**：兜底不该比上游约定更严。曾把兜底设成 3 句，
       结果回复被削到平均 2.2 句，好几轮只剩一个问句——那正是"答非所问"的来源。

    【句级安全豁免】
        原先是"出现任一受保护词就整段原样返回"，导致问号/句数/说教三条约束同时失效
        （实测出现过 2 个问句一起发出去）。现在只保护 ``policy.protected_predicate``
        判定为 True 的句子：这类句子不计入问号上限、截断时优先保留，其余照常修剪。
    """
    policy = policy or GuardPolicy()
    raw = (text or "").strip()
    info: dict = {"actions": [], "sentences_before": len(split_sentences(raw))}

    if not raw:
        info["actions"].append("empty input -> returned as is")
        info["sentences_after"] = 0
        return "", info
    if is_protected:
        info["actions"].append("protected reply -> returned unchanged (no trimming)")
        info["sentences_after"] = info["sentences_before"]
        return raw, info
    if has_trusted_number(raw, policy):
        info["actions"].append("trusted number present -> returned unchanged")
        info["sentences_after"] = info["sentences_before"]
        return raw, info

    cleaned = strip_marks(raw)
    if cleaned != raw:
        info["actions"].append("removed markdown/bullet symbols")

    sents = split_sentences(cleaned)
    kept = sents

    if policy.drop_prescriptive:
        no_lecture = [s for s in sents if not any(m in s for m in PRESCRIPTIVE_MARKERS)]
        if no_lecture and len(no_lecture) != len(sents):
            info["actions"].append(
                f"dropped {len(sents) - len(no_lecture)} prescriptive sentence(s)")
        kept = no_lecture or sents        # 全被删光时保留原句，绝不返回空

    if policy.max_questions is not None:
        q_idx = [i for i, s in enumerate(kept) if any(m in s for m in QUESTION_MARKS)]
        guard_q = [i for i in q_idx if _is_protected(kept[i], policy)]
        normal_q = [i for i in q_idx if i not in set(guard_q)]
        if guard_q:
            info["actions"].append(
                f"kept {len(guard_q)} protected question(s) outside the question cap")
        if len(normal_q) > policy.max_questions:
            drop = set(normal_q[policy.max_questions:])
            info["actions"].append(
                f"dropped {len(drop)} extra question(s) "
                f"(at most {policy.max_questions} per reply)")
            kept = [s for i, s in enumerate(kept) if i not in drop] or kept

    if len(kept) > policy.max_sentences:
        # 挑选保留哪几句：**受保护句 > 含问号的句子 > 其余按原顺序**。
        # 只挑不编、不改顺序、不改字面。
        keep_idx = [0]
        for i, s in enumerate(kept):
            if i not in keep_idx and _is_protected(s, policy):
                keep_idx.append(i)
        if len(keep_idx) > 1:
            info["actions"].append("kept protected sentence(s) when truncating")
        q_first = [i for i, s in enumerate(kept) if any(m in s for m in QUESTION_MARKS)]
        if q_first and q_first[0] != 0 and q_first[0] not in keep_idx:
            keep_idx.append(q_first[0])
            info["actions"].append("kept the question sentence when truncating")
        for i in range(1, len(kept)):
            if len(keep_idx) >= policy.max_sentences:
                break
            if i not in keep_idx:
                keep_idx.append(i)
        keep_idx = sorted(keep_idx)[: policy.max_sentences]
        info["actions"].append(
            f"truncated to {policy.max_sentences} sentence(s) (had {len(kept)})")
        kept = [kept[i] for i in keep_idx]

    out = "".join(kept).strip()
    info["sentences_after"] = len(split_sentences(out))
    if not info["actions"]:
        info["actions"].append("no change needed")
    return out, info
