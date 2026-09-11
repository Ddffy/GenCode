"""Parser for GenCode's text model protocol."""

import json
import re


def parse(raw):
    raw = str(raw)
    if "<tool" in raw and (
        "<final>" not in raw or raw.find("<tool") < raw.find("<final>") #判断是否可能是工具调用 ，文本出现了tool，且在final之前
    ):  #如果先final再tool 优先认为任务已经结束
        parsed = parse_tool_blocks(raw)  #解析工具快
        if isinstance(parsed, str):      # 说明模型输出了工具结构，但格式错误，需要重新请求模型
            return "retry", retry_notice(parsed)
        if parsed:   #如果解析出了工具，单个工具返回 "tool"，多个工具返回 "tools"。
            return _tool_kind(parsed)

    if "<final>" in raw:  #解析最终答案
        return "final", extract(raw, "final") #这里使用 extract() 去掉标签，并对内容执行 strip()

    if not raw.strip(): #空响应会被视为需要重试，而不是直接结束任务。没有 <tool> 或 <final>，GenCode 会要求模型重新按照协议输出。
        return "retry", retry_notice("empty response")
    return "retry", retry_notice("missing <tool> or <final> tag")

#重试提示函数
def retry_notice(problem=None):
    detail = f" Problem: {problem}." if problem else ""
    return (
        "Your previous response could not be executed."
        f"{detail} Return one or more valid <tool> calls, or one <final> answer."
    )

#工具参数归一化
def normalize_tool_payload(payload):
    if isinstance(payload, list):
        if not payload:
            return "tool JSON list must not be empty"
        normalized = []
        for item in payload:
            parsed = normalize_tool_payload(item)
            if isinstance(parsed, str):
                return parsed
            normalized.extend(parsed)
        return normalized
    if not isinstance(payload, dict) or "name" not in payload:
        return "tool JSON must be an object with name and args"
    args = payload.get("args", {})
    if not isinstance(args, dict):
        return "tool args must be an object"
    return [{"name": payload["name"], "args": args}]

#它支持两种写法：
# 1. XML 属性形式；
# 2. <tool> 内部嵌 JSON 形式。
def parse_tool_blocks(raw):
    tools = []
    errors = []
    for match in re.finditer(  #正则匹配所有tool。匹配<tool ...>...</tool>
        # attrs 是开始标签中的属性； <tool name="write_file" path="a.py">
                                 # <content>
                                 # print("hello")
                                 # </content>
                                 # </tool>
        # body 是开始标签和结束标签之间的内容；
        # DOTALL 允许 body 跨多行
        r"<tool\b(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", str(raw), flags=re.DOTALL
    ):
        attrs = parse_attrs(match.group("attrs"))
        if attrs.get("name", "").strip():
            parsed_xml = parse_xml_tool_match(match) # 有 name 属性时，按 XML 工具解析
            if parsed_xml:
                tools.append(parsed_xml)
            continue
        body = match.group("body").strip()
        try:                                         # 没有 name 属性时，按 JSON 解析
            payload = json.loads(body)
        except json.JSONDecodeError:
            errors.append("tool payload must be valid JSON or supported XML")
            continue
        parsed_json = normalize_tool_payload(payload)  #归一化JSON工具
        if isinstance(parsed_json, str):
            errors.append(parsed_json)
            continue
        tools.extend(parsed_json)
    if tools:
        return tools
    if errors:
        return errors[0]
    return []


def _tool_kind(tools):
    if len(tools) == 1:
        return "tool", tools[0]
    return "tools", tools


def parse_xml_tools(raw):
    tools = []
    for match in re.finditer(
        r"<tool\b(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", str(raw), flags=re.DOTALL
    ):
        parsed = parse_xml_tool_match(match)
        if parsed:
            tools.append(parsed)
    return tools


def parse_xml_tool(raw):
    match = re.search(
        r"<tool\b(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", str(raw), flags=re.DOTALL
    )
    if not match:
        return None
    return parse_xml_tool_match(match)


def parse_xml_tool_match(match):
    attrs = parse_attrs(match.group("attrs"))
    body = match.group("body")
    name = attrs.get("name", "").strip()
    if not name:
        return None
    args = {key: value for key, value in attrs.items() if key != "name"}
    for tag in ("content", "old_text", "new_text"):
        value = extract_raw(body, tag)
        if value is not None:
            args[tag] = value
    if name == "write_file" and "content" not in args and body.strip():
        args["content"] = body
    return {"name": name, "args": args}


def parse_attrs(text):
    attrs = {}
    for key, value in re.findall(
        r'([A-Za-z_][A-Za-z0-9_-]*)="(.*?)"', text, flags=re.DOTALL
    ):
        attrs[key] = value
    return attrs


def extract(text, tag):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if not match:
        return text.strip()
    return match.group(1).strip()


def extract_raw(text, tag):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1)
