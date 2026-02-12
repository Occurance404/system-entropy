# Literature Survey (Thesis Prep)

This draft summarizes only the papers we actually need for the thesis arguments. Sources are grouped by theme and aligned to the keep-list in `thesis/literature/literature-key-findings.md`.

## Agent Failures and Silent Errors
Chen2024ToolsFail shows how tool-use mistakes can stay hidden; Chen2023ToolEmu finds 68.8% of simulated failures would occur in real tool use and even “safe” agents fail 23.9% of the time. Zhou2024ToolSandbox and Fu2024Tau add stateful, multi-turn tool benchmarks that surface dependence on hidden state. Grattafiori2025MCPMark stresses realistic CRUD-heavy tasks (gpt-4-medium only 52.56% pass@1). Jiang2025DarkPatterns shows agents miss manipulative GUIs and push ahead anyway. Together these anchor the claim that agents fail quietly and benchmarks must be stateful to expose it.

## RLHF Miscalibration and Confidence
Chen2025Overconfidence documents RLHF-induced suppression of hesitation; Tian2023Calibration shows verbalized confidences beat token probabilities but both are shaky after RLHF. Perez2022Discovering and Perez2024Mislead tie RLHF to inverse scaling and persuasive-but-wrong behavior. Wang2024StyleOutweighs shows reward models prefer style over substance, while Li2024BeyondScalar and Dong2025CHARM propose calibration fixes. These support the thesis stance that token entropy is decoupled from true uncertainty in RLHF models.

## Semantic Uncertainty and Consistency
Zhou2024Consistency finds consistency-based UQ outperforms reliability-based methods, directly paralleling branching probes. Barber2022ConformalPrediction gives distribution-free foundations for non-parametric uncertainty sets; Blei2019Model formalizes comparing embedding clusters; Chen2025Ensemble shows ensemble embeddings improve semantic similarity robustness. Collectively they justify embedding-space SCR as a semantic stability signal.

## Context Management and Long-Horizon Stability
Wang2025ComplexityTrap shows simple observation masking rivals summarization, implying context rot is structural. Chen2024HiAgent’s hierarchical memory doubles success and cuts steps; Zhang2025ContextFolding reduces active context 10×; Zhou2025MemTool and Guo2024MOSS show dynamic tool/memory management. These back the need for resets/rescue and context hygiene around SCR spikes.

## Dynamic Benchmarks and Stress Tests
Jimenez2024SWEBenchPlus and Wang2025SavingSWE reveal 32.67% solution leakage and 31.08% weak tests in SWE-bench, dropping true resolution to ~3.97%—static benchmarks overstate capability. ToolSandbox, τ-bench, and MCPMark demonstrate multi-turn, stateful stress testing; Li2024SWEAgent shows interface design matters for agent-computer work. This motivates the framework’s scenario-based, perturbation-heavy design.

## Sandbox and Monitoring (Optional)
Li2025RedTeamCUA introduces hybrid web/OS adversarial sandboxes; Guo2025AgentGuard targets unsafe tool orchestration. These inform future extensions of the monitoring stack but are secondary to the core thesis.

## Takeaways for This Thesis
- Silent failure is documented and common; we lean on ToolsFail/ToolEmu + stateful benchmarks to motivate SCR.
- RLHF warps outward confidence; entropy alone is unreliable (Overconfidence, Mislead, StyleOutweighs).
- Semantic divergence methods (Consistency, conformal/UQ, ensemble embeddings) justify SCR as the primary stability metric.
- Long-horizon degradation is structural; context management and rescue protocols are necessary (HiAgent, ContextFolding, ComplexityTrap).
- Static benchmarks overestimate agents; scenario-based shocks and stateful sandboxes provide realistic evaluation (SWE-bench+, ToolSandbox, MCPMark, τ-bench).

## Extended Paper List (as provided for first review)
- Agent Failures / Silent Errors: Chen2024ToolsFail; Chen2023ToolEmu; Zhou2024ToolSandbox; Grattafiori2025MCPMark; Fu2024Tau; Jiang2025DarkPatterns.
- RLHF Miscalibration / Confidence: Chen2025Overconfidence; Tian2023Calibration; Perez2022Discovering; Perez2024Mislead; Wang2024StyleOutweighs; Li2024BeyondScalar; Dong2025CHARM.
- Semantic Uncertainty / Consistency: Zhou2024Consistency; Barber2022ConformalPrediction; Blei2019Model; Chen2025Ensemble.
- Context Management / Long-Horizon Stability: Wang2025ComplexityTrap; Chen2024HiAgent; Zhang2025ContextFolding; Zhou2025MemTool; Guo2024MOSS.
- Dynamic Benchmarks / Stress Tests: Wang2025SavingSWE; Jimenez2024SWEBenchPlus; Li2024SWEAgent; plus ToolSandbox, τ-bench, MCPMark for multi-turn/stateful evaluation.
- Sandbox / Monitoring (Optional): Li2025RedTeamCUA; Guo2025AgentGuard.
- Keep-list priority reminder: Tier 1 (Overconfidence, ToolEmu, SWE-bench+ / SavingSWE, HiAgent, Consistency); Tier 2 (StyleOutweighs, Mislead, ComplexityTrap, ContextFolding, MCPMark); Tier 3 (ConformalPrediction, RedTeamCUA, ToolSandbox, Tau, SWEAgent).
