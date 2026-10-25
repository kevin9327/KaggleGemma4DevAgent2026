# Kaggle × Google DeepMind — `The Gemma 4 Developer Agent Competition` (`$75,000 USD` Cash Pool + `$10,000 USD` Small Model Bonus Package)

![Banner](banner.png)

## 1. Competition Overview & Live Cash Verification
- **Competition:** Google — `The Gemma 4 Developer Agent Competition`
- **Official URL:** `https://www.kaggle.com/competitions/gemma-4-developer-agent`
- **Cash Prize Pool:** **`$75,000 USD` Cash (`$65,000 USD` Main Prediction Leaderboard + `$10,000 USD` Small Model Development Bonus)** + `$35,000 USD` Paper Track (`$110,000 USD` Total Ecosystem Pool)
- **Deadlines:**
  - Paper Track Deadline: **November 12, 2026 (23:59 UTC)**
  - Main Track Entry & Team Merger Deadline: **November 25, 2026 (23:59 UTC)**
  - Main Track Final Submission Deadline: **December 2, 2026 (23:59 UTC)**

---

## 2. Strategic Edge: Qualifying for the `$10,000 USD` Small Model Bonus (`Gemma E4B`)
Per the official competition rules, the **`$10,000 USD` Small Model Development Bonus** is awarded to the highest-ranking submission that uses **`Gemma E2B` (`gemma-4-e2b-it`) or `Gemma E4B` (`gemma-4-e4b-it`)** as its base model. While `Gemma4AgentPaper/` targeted the `$35,000` Paper Track with `gemma-4-31b-it-qat-w4a16-ct`, [`KaggleGemma4DevAgent2026/`](.) pins **`gemma-4-e4b-it`** uniformly across the root coordinator and all 3 sub-agents inside [`submission_e4b.zip`](submission_e4b.zip), making it simultaneously eligible for:
1. **`$10,000 USD` Small Model Development Bonus (`Gemma E2B / E4B`)**
2. **`$65,000 USD` Main Prediction Competition Leaderboard**

---

## 3. Package Files & Deterministic Verification (`exit code 0`)
- [`gemma4_e4b_swe_agent.py`](gemma4_e4b_swe_agent.py) — AST Call-Graph Subgraph Fault Localizer, Unified Diff Patch Synthesizer, Sandbox Unit Test Verifier (`4/4` SWE tasks passed), and `submission_e4b.zip` packager.
- [`submission_e4b.zip`](submission_e4b.zip) — Google ADK declarative multi-agent package (`agent.yaml`, `eval_config.yaml`, `sub_agents/*.yaml`).
- [`gemma4_e4b_verification_report.json`](gemma4_e4b_verification_report.json) — Machine-verifiable JSON report (`SHA-256: 189ef744f8ddaafb4d974df3eb4f2c6be9aa10aba2d3441195ea6e3efbceca38`).
- [`index.html`](index.html) — Interactive `Gemma-4-E4B` AST Subgraph & Unified Diff Workbench.
- [`banner.png`](banner.png) — `1200x630` competition banner.

```bash
python KaggleGemma4DevAgent2026/gemma4_e4b_swe_agent.py
```
