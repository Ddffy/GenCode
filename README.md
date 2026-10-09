<p align="center">
  <a href="https://github.com/Ddffy/GenCode/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Ddffy/GenCode?style=flat-square"></a>
  <a href="pyproject.toml"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white"></a>
</p>

# GenCode

**From Prompt to Verified Patch**

GenCode 是一个面向长周期代码任务的本地优先 Agent Harness。它把仓库理解、模型与工具执行、上下文管理、Goal 协作、Git 变更恢复和验收证据放在同一条运行链路中。

普通请求由异步 Runtime 推进；需要拆解和并行探索的长任务可以通过 `/goal` 启动持久化 DAG。Worker 在隔离的 Git worktree 中完成子任务，主协调器再串行集成并验证结果。

## Getting started

需要 Python 3.10+ 和 Git：

```bash
git clone https://github.com/Ddffy/GenCode.git
cd GenCode
python -m pip install -e .
```

配置模型 Provider：

```powershell
Copy-Item .gencode.toml.example .gencode.toml
```

macOS / Linux：

```bash
cp .gencode.toml.example .gencode.toml
```

编辑 `.gencode.toml`，填写 Provider、模型和 API key，然后启动：

```bash
gencode
```

也可以执行单次任务或启动行式 REPL：

```bash
gencode "找出测试失败的根因并修复"
gencode --repl
gencode --resume latest
```

Repo Map 和向量检索依赖可选安装：

```bash
python -m pip install -e ".[map,rag]"
```

完整选项见 `gencode --help`；配置示例见 [`.gencode.toml.example`](.gencode.toml.example)。

## What GenCode does

- **运行时与工具循环**：`Runtime` 持有会话和任务状态，`Engine` 推进异步 ReAct、原生工具调用、流式事件、预算和取消控制。
- **仓库理解与检索**：Tree-sitter Repo Map 提供预算内的结构概览；需要定位具体代码片段或知识时，可使用稀疏/稠密混合检索与重排。
- **上下文与记忆**：按预算组装当前请求、历史和工作记忆；Skill、Wiki、Spec 作为类型化长期知识检索。Dream 可从运行记录提出候选，候选经检查和确认后再进入默认召回。
- **长任务协作**：`/goal <目标>` 创建持久化 DAG；Worker attempt 使用独立 worktree，失败可重派，变更由主协调器串行集成。
- **验证与回溯**：Run trace、工具证据、verifier 结果和报告保存在本地 `.gencode/`，可用于调试、恢复和评测。

## Goal mode

Goal 的执行顺序是：固定干净工作区的 base commit → 规划并校验 DAG → 在独立 worktree 执行 Worker → 串行集成到本地分支 → 运行 verifier 和只读 Critic。合并结果是供用户审查的本地候选，不会自动推送到远端。若工作区存在未提交改动，Goal 会拒绝启动，要求先处理工作区状态。

在 TUI / REPL 中使用 `/goal <目标>` 启动；通过 `/goal status`、`/goal wait`、`/goal resume <id>` 查看或恢复任务。命令和生命周期见 `gencode/core/runtime/goal_manager.py`。

## Permissions and recovery

工具执行经过统一的参数、路径、权限和策略检查。`run_shell` 可配置 Sandbox，但示例配置默认关闭；普通文件工具的工作区范围治理不等同于 Docker 或 VM 级隔离。只在可信仓库中运行，或先配置并验证所需的隔离后端。

GenCode 在 Git 仓库中记录自身产生的变更，并支持失败后的定向撤销；它不会用全局 reset 覆盖用户原有未提交内容。Session、事件和运行证据默认保存在本地 `.gencode/`，Provider 请求会将必要的 Prompt、代码片段和工具结果发送到配置的模型服务。

## Project layout

| Package | Responsibility |
| --- | --- |
| `gencode/core/runtime` | Runtime、Engine、TaskState、Goal、Worker 与 Session 生命周期 |
| `gencode/core/context` | Prompt 组装、预算、Compact、历史与知识注入 |
| `gencode/core/actions` | 工具执行边界、权限/策略、并发和 Git 集成 |
| `gencode/features` | Repo Map、检索和 Skill / Wiki / Spec |
| `gencode/evaluation` | Harness benchmark、专项评测与指标 |

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -q
ruff check gencode tests
```

确定性 Harness benchmark 使用固定任务、Scripted Model 和独立 verifier；真实模型实验需要单独配置 Provider 凭据，二者应分别解读。
