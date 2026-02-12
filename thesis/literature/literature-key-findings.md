# Literature Anchors & Key Findings (for Thesis Write-Up)

## Agent Failures / Silent Errors
- **Chen2024ToolsFail** — Silent tool-use errors; framework detects incorrect tool calls without surfaced uncertainty.  
  *Source:* Chen et al., 2024, "Tools Fail: Detecting Silent Errors in Faulty Tools," arXiv:2406.19228.
- **Chen2023ToolEmu** — 68.8% of simulated failures valid in real world; “safest” agent still fails 23.9% of the time.  
  *Source:* Chen et al., 2023/2024, "Identifying the Risks of LM Agents with an LM-Emulated Sandbox," arXiv:2309.15817.
- **Zhou2024ToolSandbox** — Stateful tool-use benchmark with implicit state dependencies and dynamic evaluation.  
  *Source:* Zhou et al., 2024, "ToolSandbox," arXiv:2408.04682.
- **Grattafiori2025MCPMark** — 127 tasks; gpt-4-medium reaches 52.56% pass@1.  
  *Source:* Grattafiori et al., 2025, "MCPMark," arXiv:2509.24002.
- **Fu2024Tau** — Tool-agent-user interaction benchmark in dynamic domains.  
  *Source:* Fu et al., 2024, "τ-bench," arXiv:2406.12045.
- **Jiang2025DarkPatterns** — Agents miss manipulative GUI patterns; prioritize task completion over protection.  
  *Source:* Jiang et al., 2025, "Dark Patterns Meet GUI Agents," arXiv:2509.10723.

## RLHF Miscalibration / Confidence
- **Chen2025Overconfidence** — RLHF distorts confidence; models suppress hesitation despite uncertainty.  
  *Source:* Chen et al., 2025, "Taming Overconfidence in LLMs," arXiv:2410.09724.
- **Tian2023Calibration** — Verbalized confidences better than conditional probabilities, but both problematic post-RLHF.  
  *Source:* Tian et al., 2023, "Just Ask for Calibration," EMNLP 2023.
- **Perez2022Discovering** — More RLHF can worsen behaviors (inverse scaling; overconfidence, undesirable goals).  
  *Source:* Perez et al., 2022, "Discovering Language Model Behaviors with Model-Written Evaluations," arXiv:2212.09251.
- **Perez2024Mislead** — RLHF improves persuasiveness, not correctness; decouples surface fluency from task coherence.  
  *Source:* Perez et al., 2024, "Language Models Learn to Mislead Humans via RLHF," arXiv:2409.12822.
- **Wang2024StyleOutweighs** — LLM judges prefer style over substance; reward models encode surface biases.  
  *Source:* Wang et al., 2024, "Style Outweighs Substance," arXiv:2409.15268.
- **Li2024BeyondScalar / Dong2025CHARM** — Reward model miscalibration on superficial features; CHARM calibrates with Arena scores.  
  *Sources:* Li et al., 2024, "Beyond Scalar Reward Model," arXiv:2410.03742; Dong et al., 2025, "CHARM," arXiv:2504.10045.

## Semantic Uncertainty / Consistency
- **Zhou2024Consistency** — Consistency-based calibration beats reliability-based for local UQ; aligned with branching probes.  
  *Source:* Zhou et al., 2024, "Consistency Calibration," arXiv:2410.12295.
- **Barber2022ConformalPrediction** — Distribution-free uncertainty sets; formal grounding for non-parametric divergence.  
  *Source:* Barber et al., 2022, "A Gentle Introduction to Conformal Prediction," arXiv:2107.07511.
- **Blei2019Model** — Probabilistic comparison for semantic grouping of embedding clusters.  
  *Source:* Blei et al., 2019, "Model Comparison for Semantic Grouping," arXiv:1904.13323.
- **Chen2025Ensemble** — Ensemble embeddings improve similarity/caching; 92% cache hit ratio.  
  *Source:* Chen et al., 2025, "Ensemble Embedding Approach," arXiv:2507.07061.

## Context Management / Long-Horizon Stability
- **Wang2025ComplexityTrap** — Simple masking ≈ summarization efficiency; context rot is structural.  
  *Source:* Wang et al., 2025, "The Complexity Trap," arXiv:2508.21433.
- **Chen2024HiAgent** — Hierarchical memory doubles success, reduces steps by ~3.8.  
  *Source:* Chen et al., 2024, "HiAgent," arXiv:2408.09559.
- **Zhang2025ContextFolding** — 10× reduction in active context with maintained performance.  
  *Source:* Zhang et al., 2025, "Scaling Long-Horizon LLM Agent via Context-Folding," arXiv:2510.11967.
- **Zhou2025MemTool / Guo2024MOSS** — Dynamic memory/tool management for multi-turn agents.  
  *Sources:* Zhou et al., 2025, "MemTool," arXiv:2507.21428; Guo et al., 2024, "MOSS," arXiv:2409.16120.

## Dynamic Benchmarks / Stress Tests
- **Wang2025SavingSWE / Jimenez2024SWEBenchPlus** — SWE-bench leakage: 32.67% solution leakage, 31.08% suspicious patches; true resolution drops to 3.97% after filtering.  
  *Sources:* Wang et al., 2025, "Saving SWE-Bench," arXiv:2510.08996; Jimenez et al., 2024, "SWE-Bench+," arXiv:2410.06992.
- **Li2024SWEAgent** — Agent-computer interfaces enable automated SWE; interface design reduces confusion.  
  *Source:* Li et al., 2024, "SWE-agent," arXiv:2405.15793.
- **Fu2024Tau / Zhou2024ToolSandbox / Grattafiori2025MCPMark** — Multi-turn/tool stress tests (dynamic interaction, stateful tool use, realistic MCP tasks).  
  *Sources:* Fu et al., 2024, arXiv:2406.12045; Zhou et al., 2024, arXiv:2408.04682; Grattafiori et al., 2025, arXiv:2509.24002.

## Sandbox / Monitoring (Optional)
- **Li2025RedTeamCUA** — Hybrid web/OS adversarial sandbox for computer-use agents.  
  *Source:* Li et al., 2025, arXiv:2505.21936.
- **Guo2025AgentGuard** — Safety evaluation of tool orchestration.  
  *Source:* Guo et al., 2025, arXiv:2502.09809.

## Recommended Keep-List (Prioritized)
- Tier 1: `Chen2025Overconfidence`, `Chen2023ToolEmu`, `Jimenez2024SWEBenchPlus` (or `Wang2025SavingSWE`), `Chen2024HiAgent`, `Zhou2024Consistency`.
- Tier 2: `Wang2024StyleOutweighs`, `Perez2024Mislead`, `Wang2025ComplexityTrap`, `Zhang2025ContextFolding`, `Grattafiori2025MCPMark`.
- Tier 3: `Barber2022ConformalPrediction`, `Li2025RedTeamCUA`, `Zhou2024ToolSandbox`, `Fu2024Tau`, `Li2024SWEAgent`.
