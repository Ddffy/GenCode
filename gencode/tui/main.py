from __future__ import annotations

import sys

from gencode.cli import build_agent, build_arg_parser
from gencode.tui.app import GenCodeTuiApp


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.prompt:
        print("gencode-tui does not accept one-shot prompts; start the TUI and type there.", file=sys.stderr)
        return 2
    agent = build_agent(args)
    GenCodeTuiApp(agent).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
