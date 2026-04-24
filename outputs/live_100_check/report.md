# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: live
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.99 | 1.0 | 0.01 |
| Avg attempts | 1 | 1.01 | 0.01 |
| Avg token estimate | 382.59 | 388.84 | 6.25 |
| Avg latency (ms) | 3936.75 | 3422.37 | -514.38 |

## Failure modes
```json
{
  "react": {
    "none": 99,
    "wrong_final_answer": 1
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
