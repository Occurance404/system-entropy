CHAPTER 1 — INTRODUCTION

1.1 When AI Agents Fail Without Warning

Benchmarks like SWE-bench and HumanEval create a sense that LLMs are steady software helpers (Jimenez2024SWEBenchPlus; OpenAI2024HumanEval). That confidence cracks as soon as the task changes mid-stream. In my runs, once the plan was formed, a new instruction did not trigger hesitation; the agent just kept executing the old plan—politely acknowledging the update while doing the wrong thing. The output stayed fluent and tidy, but it marched in the wrong direction.

This “silent failure” repeats whenever a plan must be revised (Chen2024ToolsFail; Chen2023ToolEmu; Jiang2025DarkPatterns). Token-level signals stay calm, so common monitors do not flag the drift until the agent is already looping or corrupting state.

1.2 Why Existing Confidence Metrics Fall Short

For years, token-level entropy has been treated as a useful proxy for model confidence. The logic is straightforward: if a model is unsure, its probability distribution over the next token spreads out; if it is confident, the distribution concentrates.

This intuition breaks down in RLHF-trained models (Chen2025Overconfidence; Perez2024Mislead; Wang2024StyleOutweighs).

Human preference rewards confident, decisive-sounding responses. During RLHF fine-tuning, models learn to suppress hesitation, even in situations where internal uncertainty is high. As a result, the entropy of the output distribution becomes decoupled from the model’s actual cognitive state. In practice:

a model may be deeply confused internally

yet still emit low-entropy, high-certainty text

because that is what it has been trained to do (Tian2023Calibration)

This distortion means that monitoring entropy alone is misleading. In my experiments, entropy often remained at or near zero even when the agent had completely lost the thread of the task. The model will not warn you when it is confused; RLHF has taught it to sound composed no matter what.

1.3 A Different Lens: Confusion Reveals Itself Through Divergent Plans

While entropy fails to reflect confusion, something else reliably does: the diversity of the model’s internal possibilities.

The intuition came from watching humans: a confident engineer states one clear next step; a confused one rattles off competing options. LLMs act the same way internally, but the final message hides it. If we ask for several possible next actions from the same state (turning up temperature a bit), we can see whether those branches agree or diverge (Zhou2024Consistency).

This led to the design of Branching Probes, a simple but revealing mechanism:

Freeze the agent’s context at a given step

Generate five parallel next-step responses

Embed each response into vector space

Measure how far apart they are

The resulting metric—the Semantic Collapse Ratio (SCR)—quantifies how fractured the agent’s internal reasoning has become. High SCR consistently appears at the exact moment the agent loses coherence.

1.4 What This Framework Tries to Solve

Because this collapse is silent in the traditional sense, we need a more direct way to measure the stability of an agent’s internal reasoning. The Entropic Stress-Test Framework was developed precisely for this purpose. Instead of relying on the model’s self-reported confidence, the framework evaluates models under dynamic, shifting task constraints—much closer to real engineering conditions.

The system consists of four tightly-coordinated components:

The Orchestrator — manages execution, introduces requirement changes, runs branching probes

The Agent Wrapper — exposes a uniform interface for any LLM and captures tool-use actions

The Sandbox — a Docker-based environment where actions have persistent consequences

The Metric Service — computes SCR, token entropy, information gain, debt indices, and loop signatures

Together, these components provide a controlled but realistic arena for observing how an agent behaves when its assumptions are challenged.

1.5 Stress-Testing in Three Practical Domains

The scenarios mimic everyday tasks a developer or data engineer might do. Each has a stable “golden path,” then a mid-run change that forces the agent to rethink.

The three primary scenarios used in this study were:

Drug Filter Shock (Data Science)

Task: Filter chemical data and refine selection criteria

Shock: Replace a simple column lookup with an external API call

Tests: Flexibility and willingness to modify working code

Directory Organizer Shock (System Operations)

Task: Clean up a cluttered directory by grouping files

Shock: Change the sorting rule mid-task

Tests: Inhibition of actions already underway

Log Parsing Shock (ETL Pipeline)

Task: Extract structured records from semi-structured logs

Shock: Alter schema assumptions in the middle of parsing

Tests: Ability to refactor logic under new constraints

These scenarios are deliberately lightweight on the surface, yet they reveal deep structural vulnerabilities when requirements shift.

1.6 Early Observations from Running These Tests

A clear, repeated pattern emerged across all experiments (Chen2024ToolsFail; Chen2023ToolEmu):

SCR spikes the moment a requirement changes, indicating internal confusion

token entropy remains low, creating an illusion of confidence

agents often fall into repetitive stagnation loops, repeatedly executing the same failing tool action

context becomes polluted, causing the agent to rely on outdated instructions

This combination of hidden confusion and outward certainty is the central challenge for building dependable autonomous agents.

1.7 Why These Findings Matter for Real Systems

In production, silent failure is worse than visible failure. The agent keeps going, sounds confident, and can corrupt state before anyone notices. The issue is not that models fail—failure is expected—it is that they fail quietly and repetitively.

The work in this thesis aims to shift the conversation from “Can the model solve a benchmark?” to “Can the model maintain coherence when the world changes around it?” The latter is much closer to the reality of production systems.

Moreover, this research provides a concrete tool—SCR—for detecting collapse early enough to intervene. This opens the door to dynamic model handoff architectures, where a stronger or more capable agent is brought in only when collapse is detected. This has immediate implications for cost, reliability, and system design.

1.8 How This Thesis Is Organized

To keep the narrative clear, the remainder of the thesis is structured as follows:

Chapter 2 formalizes the theoretical underpinnings of cognitive collapse and explains why token-level metrics are insufficient.

Chapter 3 presents the complete system design, including architectural diagrams, metric definitions, and the branching-probe algorithm.

Chapter 4 describes the experimental setup and presents results across all stress-testing scenarios.

Chapter 5 interprets these findings and discusses implications for real-world agent design.

Chapter 6 concludes with limitations and future research directions.

1.9 Main Contributions

This thesis makes four contributions:

It identifies and characterizes a recurring silent failure mode in RLHF-trained agents under dynamic task conditions.

It introduces Semantic Collapse Ratio (SCR), a practical and theoretically grounded metric for evaluating internal divergence in agent reasoning.

It presents a reproducible framework for stress-testing agents in multi-step, non-stationary environments.

It demonstrates empirically that SCR provides strong early-warning signals that entropy-based methods miss entirely.
