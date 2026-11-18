# -*- coding: utf-8 -*-
"""guard 的回归用例。

【这份测试测什么】
    只测规则本身：几何约束（句数/问号）、只删不编、绝不返回空、以及五条
    「规则太宽会误删正常内容」的历史回归。

【为什么每条规则都要有反例用例】
    这个模块里真正的风险不是"没生效"，而是"生效得太狠"：
    判据放宽一点，正常句子就被当垃圾删掉，而用户看到的是答非所问——比报错更难发现。
    每条规则都配一个"不该动它"的用例，比只写正向用例值钱。

"受保护句"在测试里用 ``★`` 前缀表示：实际使用中由调用方的
``GuardPolicy.protected_predicate`` 决定，这里用一个抽象标记避免依赖业务语义。
（标记不能选句末标点——``!``/``?`` 会被切句规则当成句子边界。）
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guard import (  # noqa: E402
    GuardPolicy,
    drop_redundant_questions,
    drop_repeated_sentences,
    enforce,
    question_count,
    split_sentences,
    strip_foreign_words,
    strip_marks,
    strip_repeated_opener,
    strip_report_frame,
    strip_untrusted_numbers,
    untrusted_numbers,
)

TRUSTED = frozenset({"555-0100"})
PROTECTED_MARK = lambda s: s.strip().startswith("★")  # noqa: E731


# ---------------------------------------------------------------- 几何约束


def test_sentence_cap_truncates_and_keeps_order():
    text = "第一句。第二句。第三句。第四句。第五句。"
    out, info = enforce(text, GuardPolicy(max_sentences=4))
    assert len(split_sentences(out)) == 4
    assert out.startswith("第一句。第二句。")
    assert "truncated" in " ".join(info["actions"])


def test_sentence_cap_keeps_the_question_sentence():
    # 截断时若把唯一的问句丢掉，回复就变成纯粹的自言自语
    text = "背景一。背景二。背景三。背景四。你想从这里开始吗？"
    out, _ = enforce(text, GuardPolicy(max_sentences=2))
    assert question_count(out) == 1


def test_question_cap_leaves_at_most_one_question():
    text = "听起来不太好。你想说说吗？还有别的吗？最在意哪个？"
    out, _ = enforce(text, GuardPolicy(max_questions=1))
    assert question_count(out) == 1
    assert len(split_sentences(out)) <= 4


def test_marks_are_removed():
    text = "- 第一条。\n- 第二条。\n## 第三条。"
    out, info = enforce(text)
    assert "-" not in out and "#" not in out
    assert "removed markdown" in " ".join(info["actions"])


# ---------------------------------------------------------------- 不变量


def test_protected_reply_returns_verbatim():
    text = "请拨打 555-0100。你现在安全吗？\n- 第二点。"
    out, info = enforce(text, GuardPolicy(trusted_numbers=TRUSTED), is_protected=True)
    assert out == text
    assert "returned unchanged" in " ".join(info["actions"])


def test_trusted_number_short_circuits_every_rule():
    # 整段短路是有意的：这类文本宁可啰嗦，也不能被裁掉任何一部分
    text = ("请拨打 555-0100。你现在安全吗？身边有人吗？\n"
            "- 你准备好了吗？\n- 你应该先冷静下来。")
    out, info = enforce(text, GuardPolicy(trusted_numbers=TRUSTED, max_sentences=2))
    assert out == text
    assert "trusted number present" in " ".join(info["actions"])


@pytest.mark.parametrize("text", [
    "第一句。第二句。第三句。第四句。第五句。第六句。",
    "你想说说吗？还有别的吗？最在意哪个？",
    "- 你应该先冷静。\n- 这是第一点。\n- 这是第二点。\n- 这是第三点。\n- 这是第四点。",
    "题目不错。也 worth 记住。你想聊什么吗？",
])
def test_only_deletes_never_adds(text):
    out, _ = enforce(text)
    reference = strip_marks(text)
    for sentence in split_sentences(out):
        assert sentence in reference, f"输出出现了原文没有的内容: {sentence!r}"


def test_never_returns_empty_even_if_every_rule_fires():
    # 全是说教句 → 说教规则会全删，但必须退回原句而不是空串
    text = "你应该早点睡。你必须听我的。"
    out, _ = enforce(text)
    assert out.strip() != ""


def test_empty_input_is_an_empty_string():
    out, info = enforce("   ")
    assert out == ""
    assert info["sentences_after"] == 0


# ---------------------------------------------------------------- 判据宽度（历史回归）


def test_prescriptive_sentence_is_dropped():
    text = "我理解你。你应该早点睡。你现在感觉怎么样呢？"
    out, _ = enforce(text)
    assert "你应该早点睡" not in out
    assert "你现在感觉怎么样呢？" in out


def test_common_noun_is_not_mistaken_for_prescription():
    # 「解决办法」是在描述对方自己找到的东西，不是助手在给建议 —— 整句必须保住
    text = "听到你找到了问题的解决办法，真的好开心。你现在感觉怎么样呢？"
    out, _ = enforce(text)
    assert "解决办法" in out


def test_sentence_level_dedup_drops_a_rephrased_repeat():
    previous = ["你为这个活动花的心思，真的很难让人忽视。你现在感觉怎么样？"]
    text = "你为这个活动付出的心思，真的很难让人忽视。你现在觉得最难的是什么？"
    out, info = drop_repeated_sentences(text, previous, threshold=0.80)
    assert out == "你现在觉得最难的是什么？"
    assert "previous turn" in " ".join(info["actions"])


def test_sentence_level_dedup_keeps_new_content():
    previous = ["你现在感觉怎么样？"]
    text = "报名的人少，你花的心思像没被接住。"
    out, _ = drop_repeated_sentences(text, previous, threshold=0.80)
    assert out == text


# ---------------------------------------------------------------- 开场白与汇报腔


def test_filler_opener_is_dropped():
    out, _ = strip_repeated_opener("你说得对，你现在感觉很累吧？我在这里听你讲。")
    assert out.startswith("你现在感觉很累吧？")


def test_repeated_short_opener_is_dropped():
    out, _ = strip_repeated_opener("我听到了，你为此付出了很多。你现在感觉怎么样？",
                                   ["我听到了，那我们继续吧。"])
    assert out.startswith("你为此付出了很多。")


def test_opener_with_content_is_kept():
    opener = "你为这个活动花的心思，真的很难想象。"
    out, _ = strip_repeated_opener(opener + "你现在感觉怎么样？", [opener])
    assert out == opener + "你现在感觉怎么样？"


def test_opener_that_is_the_whole_reply_is_kept():
    # 删了就没内容了 → 不许删
    out, _ = strip_repeated_opener("你说得对。", [])
    assert out == "你说得对。"


def test_report_frame_is_stripped_but_content_kept():
    out, _ = strip_report_frame("你提到的橘子，是不是一直记得呢？")
    assert out == "橘子，是不是一直记得呢？"


def test_report_frame_does_not_cut_a_normal_sentence():
    # 光板「你说」可能是正常句子的开头，剥掉会削出残句
    text = "你说话的时候情绪很重，我想听你说说。"
    out, _ = strip_report_frame(text)
    assert out == text


# ---------------------------------------------------------------- 语言与外来的号码


def test_foreign_word_sentence_is_dropped():
    out, _ = strip_foreign_words("今天天气不错，也 worth 记住。你想聊什么吗？")
    assert out == "你想聊什么吗？"


def test_foreign_word_kept_when_text_carries_a_trusted_number():
    text = "请拨打 555-0100，也可以 OK 的。"
    out, _ = strip_foreign_words(text, trusted_numbers=TRUSTED)
    assert out == text


def test_untrusted_number_sentence_is_dropped():
    text = "我理解你。请打这个电话：400-123-4567，他们会帮你。你现在感觉怎么样？"
    out, info = strip_untrusted_numbers(text)
    assert out == "我理解你。你现在感觉怎么样？"
    assert untrusted_numbers(out) == []
    assert info["dropped_numbers"]


def test_text_without_numbers_is_untouched():
    text = "听起来这段时间你扛了很多。是工作里发生了什么？"
    out, _ = strip_untrusted_numbers(text)
    assert out == text


# ---------------------------------------------------------------- 收尾轮的冗余追问


def test_redundant_confirmation_keeps_the_statement():
    out, info = drop_redundant_questions("周六下午三点见，你准备好了吗？")
    assert out == "周六下午三点见。"
    assert "confirmation" in " ".join(info["actions"])


def test_option_question_is_preserved():
    # 给选项是在把选择权交回去，不是催确认
    text = "是周六下午三点，你需要更改时间吗？"
    out, _ = drop_redundant_questions(text)
    assert out == text


def test_natural_question_is_preserved():
    text = "我记得你是小岚，最近感觉怎么样？"
    out, _ = drop_redundant_questions(text)
    assert out == text


def test_all_prompts_is_not_deleted_to_empty():
    text = "你准备好了吗？"
    out, info = drop_redundant_questions(text)
    assert out == text
    assert "never return empty" in " ".join(info["actions"])


# ---------------------------------------------------------------- 句级豁免


def test_protected_sentence_is_exempt_from_the_question_cap():
    policy = GuardPolicy(max_questions=1, protected_predicate=PROTECTED_MARK)
    text = "★必须问清的这一句？第二问？第三问？"
    out, info = enforce(text, policy)
    assert "★必须问清的这一句？" in out
    assert "protected question" in " ".join(info["actions"])


def test_protected_sentence_does_not_disable_the_other_rules():
    # 这就是「整段白名单」的反面教材：豁免必须句级，其余规则照常生效
    policy = GuardPolicy(max_sentences=3, max_questions=1,
                         protected_predicate=PROTECTED_MARK)
    text = ("★必答第一句。\n- 你应该先冷静下来。\n普通第三句。普通第四句。"
            "普通第五句。普通第六句。")
    out, _ = enforce(text, policy)
    assert "★必答第一句" in out
    assert "- " not in out            # 分点仍被清
    assert "你应该" not in out         # 说教句仍被删
    assert len(split_sentences(out)) <= 3
