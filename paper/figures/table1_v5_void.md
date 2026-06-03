# Table 1 — Main Results (Paper v5)

τ_void (primary) = 0.35

| Benchmark | Metric | B0 | B2 | A3 | B2+void | A1+void | A3+void |
|---|---|---|---|---|---|---|---|
| hotpotqa | EM | 74.0% | 70.0% | 74.0% | 74.0% | 76.0% | 74.0% |
| 2wikimultihopqa | EM | 58.0% | 56.0% | 60.0% | 58.0% | 58.0% | 58.0% |
| musique | EM | 42.0% | 40.0% | 40.0% | 42.0% | 44.0% | 44.0% |
| triviaqa | EM | 76.7% | 80.0% | 80.0% | 80.0% | 80.0% | 76.7% |
| gsm8k | EM | 60.0% | 63.3% | 63.3% | 60.0% | 66.7% | 70.0% |
| longmemeval | EM | 40.0% | 35.0% | 30.0% | 30.0% | 30.0% | 30.0% |
| locomo | F1 | 9.8% | 7.6% | 13.0% | 10.9% | 8.2% | 14.9% |
|||||||||
| **Mean** | — | **51.5%** | **50.3%** | **51.5%** | **50.7%** | **51.8%** | **52.5%** |

## Token Cost (avg per task)

| Benchmark | B0 | B2 | A3 | B2+void | A1+void | A3+void |
|---|---|---|---|---|---|---|
| hotpotqa | 0 | 647 | 326 | 11 | 11 | 7 |
| 2wikimultihopqa | 0 | 635 | 187 | 51 | 22 | 10 |
| musique | 0 | 685 | 259 | 29 | 27 | 11 |
| triviaqa | 0 | 617 | 232 | 110 | 104 | 49 |
| gsm8k | 0 | 571 | 281 | 496 | 456 | 240 |
| longmemeval | 0 | 716 | 330 | 568 | 568 | 263 |
| locomo | 0 | 702 | 252 | 702 | 586 | 252 |

## Void Rate (% of tasks routed to c_∅)

| Benchmark | B2+void | A1+void | A3+void |
|---|---|---|---|
| hotpotqa | 98.0% | 98.0% | 98.0% |
| 2wikimultihopqa | 92.0% | 92.0% | 92.0% |
| musique | 96.0% | 96.0% | 96.0% |
| triviaqa | 83.3% | 83.3% | 83.3% |
| gsm8k | 13.3% | 13.3% | 13.3% |
| longmemeval | 20.0% | 20.0% | 20.0% |
| locomo | 0.0% | 0.0% | 0.0% |