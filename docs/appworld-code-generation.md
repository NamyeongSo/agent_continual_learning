# AppWorld interaction modes

Exgentic uses AppWorld's interactive Python code-generation mode by default. The agent receives one `execute_python`
action and submits a Python code block on each turn. AppWorld executes those blocks in its persistent task shell, so
variables can be reused across turns. The shell exposes `apis` for application APIs, including `apis.api_docs` for
documentation lookup; a single block can use Python control flow and make several API calls. Print values that should be
returned to the agent, and complete a task through `apis.supervisor.complete_task(...)`.

```python
from exgentic import AppWorldBenchmark, LiteLLMToolCallingAgent, evaluate

evaluate(
    benchmark=AppWorldBenchmark(subset="test_normal"),
    agent=LiteLLMToolCallingAgent(model="gpt-4o"),
    num_tasks=3,
)
```

The benchmark exposes Python execution as one action because Exgentic's agent interface is action-based. The action's
argument is generated Python source, and the AppWorld shell executes that source; app APIs are not separately exposed
as model tools in this mode.

To reproduce older runs that exposed each AppWorld API as a separate tool, select the legacy function-calling mode:

```python
benchmark = AppWorldBenchmark(subset="test_normal", interaction_mode="function_calling")
```

The selected mode is recorded in benchmark/session configuration. Both modes retain the shared actor limits: up to
100,000 input tokens and 2,048 output tokens per turn, with at most 40 turns per trajectory. In code-generation mode,
one submitted Python block is one agent turn; API calls made within it are still subject to AppWorld's own interaction
limit.
