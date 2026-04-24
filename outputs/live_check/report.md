# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: live
- Records: 16
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 1.0 | 1.0 | 0.0 |
| Avg attempts | 1 | 1 | 0 |
| Avg token estimate | 381.75 | 382.88 | 1.13 |
| Avg latency (ms) | 4950.12 | 2827.62 | -2122.5 |

## Failure modes
```json
{
  "react": {
    "none": 8
  },
  "reflexion": {
    "none": 8
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
