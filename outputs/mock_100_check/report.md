# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: mock
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.96 | 1.0 | 0.04 |
| Avg attempts | 1 | 1.04 | 0.04 |
| Avg token estimate | 475 | 627 | 152 |
| Avg latency (ms) | 240 | 347.2 | 107.2 |

## Failure modes
```json
{
  "react": {
    "none": 96,
    "incomplete_multi_hop": 1,
    "wrong_final_answer": 1,
    "entity_drift": 2
  },
  "reflexion": {
    "none": 100
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
This benchmark compares a single-shot ReAct-style actor against a Reflexion loop that retries after structured feedback. The structured evaluator produces machine-parseable scores, reasons, missing evidence, and failure modes, which makes the retry policy deterministic and report-friendly. Reflection memory is injected only after failed attempts, so the actor can revise its next answer using concrete lessons instead of vague self-critique. The tradeoff is higher latency and token cost, but the end-to-end loop is now ready for a real LLM endpoint and real Hotpot-style examples.
