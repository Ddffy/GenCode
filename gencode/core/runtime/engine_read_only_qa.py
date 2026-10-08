READ_ONLY_QA_STEP_BUDGET = 28
READ_ONLY_QA_ATTEMPT_BUDGET = 8
READ_ONLY_QA_EVIDENCE_STEP_BUDGET = 12
READ_ONLY_QA_FINAL_MAX_NEW_TOKENS = 1800
READ_ONLY_QA_FINAL_NOTICE = (
    "Evidence budget reached. Do not call more tools; answer the original question "
    "now using the evidence already collected. Keep the complete answer under "
    "about 1200 Chinese characters, with at most five concise bullets."
)


def _is_read_only_qa_request(user_message):
    text = str(user_message or "").strip().lower()
    for negated_action in (
        "不要修改代码", "不修改代码", "无需修改代码", "不需要修改代码",
        "不要改代码", "不改代码", "do not modify code", "don't modify code",
        "without modifying code",
    ):
        text = text.replace(negated_action, "")
    question_markers = (
        "说说", "介绍", "解释", "讲讲", "详细说", "怎么", "如何", "为什么",
        "是什么", "是否", "对比", "有哪些", "what is", "how does", "explain",
        "describe", "compare",
    )
    action_markers = (
        "改一下", "修改代码", "帮我改", "帮我实现", "实现这个功能", "修复这个",
        "添加功能", "删除文件", "运行测试", "执行测试", "提交代码", "commit一下",
        "please implement", "implement this", "fix this", "change the code",
        "run tests", "commit the changes",
    )
    return any(marker in text for marker in question_markers) and not any(
        marker in text for marker in action_markers
    )


def _turn_tool_step_budget(user_message, max_steps, *, allow_read_only_qa=True):
    if allow_read_only_qa and _is_read_only_qa_request(user_message):
        return min(int(max_steps), READ_ONLY_QA_STEP_BUDGET)
    return int(max_steps)


def _read_only_qa_final_only(*, read_only_qa, tool_steps, native_mode):
    return bool(
        read_only_qa
        and native_mode
        and int(tool_steps) >= READ_ONLY_QA_EVIDENCE_STEP_BUDGET
    )


def _read_only_qa_prompt(user_message):
    return (
        f"{user_message}\n\n"
        "只读仓库问答执行约束：不要输出探索过程；先用 search(pattern, path) 定位符号，"
        "再用 read_file(path, start, end) 读取精确代码段；必须显式传 start/end，单次最多 80 行，"
        "不要整文件读取；不要重复读取同一路径，也不要读取 .gencode/runs 下的产物。"
        "不使用 Shell，不修改文件；针对跨文件问题沿实际调用链核对入口、结果回传和停止/失败路径，"
        "证据足够后立即回答。最终用中文控制在约 1000 字以内，覆盖结论、关键运行机制和文件/函数出处；"
        "区分代码中确认的事实与推断，未核实的细节明确说明。"
    )
