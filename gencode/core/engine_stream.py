"""Filter protocol markup out of streamed user-visible text."""

_CONTROL_TAGS = ("<tool>", "<tools>", "<retry>", "<final>")
TURN_STREAM_EVENTS = frozenset(
    {
        "turn_started",
        "context_building",
        "text_delta",
        "model_requested",
        "model_parsed",
        "tool_call",
        "tool_result",
        "retry",
        "runtime_notice",
        "final",
        "stop",
        "turn_finished",
    }
)


class VisibleTextStream:
    def __init__(self):
        self.buffer = ""
        self.mode = "auto"

    def feed(self, text):
        self.buffer += str(text or "")
        return self._drain(False)

    def finish(self):
        return self._drain(True)

    def _drain(self, final):
        output = []
        while self.buffer:
            if self.mode == "suppressed":
                self.buffer = ""
                break
            if self.mode == "final":
                closing = "</final>"
                index = self.buffer.find(closing)
                if index >= 0:
                    if index:
                        output.append(self.buffer[:index])
                    self.buffer = self.buffer[index + len(closing):]
                    self.mode = "suppressed"
                    continue
                safe = len(self.buffer) if final else max(0, len(self.buffer) - len(closing) + 1)
                if safe:
                    output.append(self.buffer[:safe])
                    self.buffer = self.buffer[safe:]
                break
            index = self.buffer.find("<")
            if index < 0:
                safe = len(self.buffer) if final else max(0, len(self.buffer) - 7)
                if safe:
                    output.append(self.buffer[:safe])
                    self.buffer = self.buffer[safe:]
                break
            if index:
                output.append(self.buffer[:index])
                self.buffer = self.buffer[index:]
            matched = next(
                (tag for tag in _CONTROL_TAGS if self.buffer.startswith(tag)), None
            )
            if matched:
                self.buffer = self.buffer[len(matched):]
                self.mode = "final" if matched == "<final>" else "suppressed"
                continue
            if self.buffer == "<tool" and not final:
                break
            if self.buffer.startswith(("<tool ", "<tool\t", "<tool\n", "<tool\r")):
                self.buffer = ""
                self.mode = "suppressed"
                continue
            if not final and any(tag.startswith(self.buffer) for tag in _CONTROL_TAGS):
                break
            output.append(self.buffer[0])
            self.buffer = self.buffer[1:]
        return [part for part in output if part]
