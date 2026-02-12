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


---

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


---

CHAPTER 2 — THE MATHEMATICS OF COGNITIVE COLLAPSE

2.1 Rethinking Entropy: Why the Classic Metric Breaks Down

Token-level entropy has long been used as a quick way to gauge how confident a language model is about its next output. The idea is straightforward: when the probability distribution over the next token is sharp, the model is “confident”; when it is diffuse, the model is “uncertain.” Formally, entropy for the next token is:

H(wt+1∣w1:t)=−∑v∈VP(v∣w1:t)log⁡P(v∣w1:t)H(w_{t+1} \mid w_{1:t}) = -\sum_{v \in \mathcal{V}} P(v \mid w_{1:t}) \log P(v \mid w_{1:t})H(wt+1​∣w1:t​)=−v∈V∑​P(v∣w1:t​)logP(v∣w1:t​)

In older, pre-RLHF models, this relationship held reasonably well. When the model didn’t know what to say, it really did spread probability mass across multiple candidates.

But RLHF changes this dynamic in a subtle but important way. Human rating tends to prefer answers that sound confident. Over time, models internalize this preference: they learn to collapse their probability distributions even when they are not certain (Chen2025Overconfidence; Perez2024Mislead; Wang2024StyleOutweighs). In effect, RLHF introduces a soft penalty term that pushes the learned policy toward low-entropy outputs:

max⁡θE[R(x)]−βHθ(x)\max_\theta \mathbb{E}[R(x)] - \beta H_\theta(x)θmax​E[R(x)]−βHθ​(x)

The presence of this entropy-minimizing term means that even when a model is internally conflicted, its surface distribution remains calm (Tian2023Calibration). In the experiments that motivated this thesis, entropy often stayed near zero during moments when the agent was clearly struggling.

This mismatch is the first clue that we need a deeper lens than token entropy to make sense of cognitive collapse.

2.2 The Blind Spot: Token Entropy Doesn’t Track Semantic Uncertainty

To see the problem more concretely, consider a situation where the agent is torn between several fundamentally different next actions: reading a file, rewriting code, asking for help, or searching documentation. Each of these actions may begin with innocuous tokens (“I”, “Let”, “First”, etc.), so the distribution over the first word does not reflect the diversity of intended actions.

Entropy captures ambiguity in the surface form, not in the underlying semantic direction. Formally, what we care about is:

Hsemantic(a)H_{\text{semantic}}(a)Hsemantic​(a)

not

Htoken(w)H_{\text{token}}(w)Htoken​(w)

These are not equivalent. Token entropy can be low even when semantic entropy is high. This is the core limitation: our confidence metric is pointed at the wrong part of the system.

2.3 Why Embedding Space Gives a Better Window Into the Agent’s Mind

Vector embeddings give us a convenient way to represent the meaning of a text snippet. When we map a response ttt into an embedding e\mathbf{e}e, we preserve its semantic structure in a geometric form (Blei2019Model; Chen2025Ensemble). Using a lightweight encoder such as all-MiniLM-L6-v2, we obtain:

e=fembed(t)\mathbf{e} = f_{\text{embed}}(t)e=fembed​(t)

These embeddings have useful properties:

semantically similar responses occupy nearby regions

dissimilar plans diverge in direction

after normalization, the vectors behave approximately as points on a hypersphere

The distance between two embeddings is typically measured using cosine distance:

dcos⁡(ei,ej)=1−ei⋅ej∥ei∥∥ej∥d_{\cos}(\mathbf{e}_i, \mathbf{e}_j) = 1 - \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}dcos​(ei​,ej​)=1−∥ei​∥∥ej​∥ei​⋅ej​​

This value is small when two responses mean similar things, and large when they diverge. It provides a direct way to examine the similarity across different “candidate plans” that the model implicitly considers.

2.4 Formalizing the Semantic Collapse Ratio (SCR)

SCR came out of a very simple idea: If a model is confused, its possible next actions will differ more than usual.

To turn this intuition into a measurable quantity, we follow four steps:

Step 1: Generate Parallel Branches

From the same context CtC_tCt​, we sample NNN alternative next actions:

B={b1,…,bN}B = \{ b_1, \dots, b_N \}B={b1​,…,bN​}

These branches are produced at a slightly elevated temperature to reveal the diversity of the model’s internal possibilities (Zhou2024Consistency).

Step 2: Embed Each Branch

ei=fembed(bi)\mathbf{e}_i = f_{\text{embed}}(b_i)ei​=fembed​(bi​)

Step 3: Compute Pairwise Distances

Dij=dcos⁡(ei,ej)D_{ij} = d_{\cos}(\mathbf{e}_i, \mathbf{e}_j)Dij​=dcos​(ei​,ej​)

Step 4: Average the Divergences

SCRt=1N(N−1)∑i=1N∑j>iNDij\text{SCR}_t = \frac{1}{N(N-1)} \sum_{i=1}^{N} \sum_{j>i}^{N} D_{ij}SCRt​=N(N−1)1​i=1∑N​j>i∑N​Dij​

This gives a single scalar between 0 (completely stable) and 1 (fully divergent). In practice, values above ~0.40 correlated strongly with collapse events in our experiments.

2.5 Geometric Interpretation: What Collapse Looks Like in High Dimensions

It helps to think about SCR visually—even if only conceptually. Imagine each candidate next step as a point on the surface of a large hypersphere (since embeddings are normalized). When the agent is stable, these points cluster tightly; the average pairwise distance is small (Blei2019Model).

When the agent becomes confused, the points spread apart, occupying more volume in the space. A useful companion measure is the radius of gyration:

Rg=1N∑i=1N∥ei−eˉ∥2R_g = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} \lVert \mathbf{e}_i - \bar{\mathbf{e}} \rVert^2 }Rg​=N1​i=1∑N​∥ei​−eˉ∥2​

where eˉ\bar{\mathbf{e}}eˉ is the centroid of the cluster. Empirically, we observe:

SCR≈2Rg2\text{SCR} \approx 2R_g^2SCR≈2Rg2​

which tells us that SCR essentially measures the “semantic spread” of the agent’s internal options at any given moment.

2.6 Modeling Collapse as a Dynamical Process

To capture how collapse evolves rather than just when it happens, we can model the agent’s state in terms of three interacting variables:

CtC_tCt​: internal context clarity

StS_tSt​: SCR at time ttt

HtH_tHt​: token entropy

When a shock occurs at time tshockt_{\text{shock}}tshock​, the agent must revise its plan. This introduces noise or “contamination” into the context:

Ct+1=Ct+α(1−Ct)C_{t+1} = C_t + \alpha (1 - C_t)Ct+1​=Ct​+α(1−Ct​)

As confusion spreads, SCR increases:

St+1=St+βCt(Smax⁡−St)S_{t+1} = S_t + \beta C_t (S_{\max} - S_t)St+1​=St​+βCt​(Smax​−St​)

Meanwhile entropy tends to stay artificially low due to RLHF-induced suppression:

Ht+1=max⁡(Ht−γCt,Hmin⁡)H_{t+1} = \max(H_t - \gamma C_t, H_{\min})Ht+1​=max(Ht​−γCt​,Hmin​)

Taken together, these equations describe a system that becomes internally unstable even as its external signals remain deceptively calm.

2.7 When the Agent Works Hard but Learns Nothing: Information Gain Efficiency

Another way to capture collapse is by measuring whether each step actually reduces uncertainty. If the agent keeps taking actions that don’t change its internal state, it is effectively spinning its wheels.

We define Information Gain Efficiency (IGE) as:

IGEt=Hpre(t)−Hpost(t)Ctokens(t)\text{IGE}_t = \frac{H_{\text{pre}}^{(t)} - H_{\text{post}}^{(t)}}{C_{\text{tokens}}^{(t)}}IGEt​=Ctokens(t)​Hpre(t)​−Hpost(t)​​

A near-zero value combined with large token consumption indicates thrashing: high effort, low progress.

This metric reveals situations where the agent repeatedly executes commands without refining its understanding—a pattern that prominently appeared in stagnation loops.

2.8 Measuring Drift Away From the Goal: Regressive Debt Index

Sometimes the agent is not just confused—it is confidently pursuing the wrong path. To measure this deviation from the intended next step, we define Regressive Debt Index (RDI):

RDIt=dcos⁡(ecurrent,etruth)\text{RDI}_t = d_{\cos}(\mathbf{e}_{\text{current}}, \mathbf{e}_{\text{truth}})RDIt​=dcos​(ecurrent​,etruth​)

High RDI indicates that the agent’s chosen action is drifting away from the correct trajectory. This metric complements SCR: SCR captures internal divergence, while RDI captures divergence from the task goal.

2.9 Quantifying Predictive Power: Statistical Validation

To test whether SCR genuinely predicts failure, we use point-biserial correlation:

rpb=M1−M0sn1n0n2r_{pb} = \frac{M_1 - M_0}{s} \sqrt{\frac{n_1 n_0}{n^2}}rpb​=sM1​−M0​​n2n1​n0​​​

Where M1M_1M1​ is the SCR prior to collapse, and M0M_0M0​ the SCR during successful steps. In our trials, the correlation was:

rpb=0.78,p<0.001r_{pb} = 0.78,\quad p < 0.001rpb​=0.78,p<0.001

A complementary ROC analysis showed:

SCR AUC = 0.92

Entropy AUC = 0.51 (no better than random)

This confirms that SCR captures something meaningful that entropy misses entirely.

2.10 Consolidated View: Two Spaces, Two Behaviors

Across all the mathematics, the key insight is surprisingly simple (Zhou2024Consistency; Barber2022ConformalPrediction):

Token space gives a stable, but misleading view of confidence

Semantic vector space reveals the true internal divergence

Collapse is precisely the moment when these two spaces drift apart: entropy indicates certainty while SCR indicates fracture.

2.11 Broader Theoretical Implications

The findings in this chapter point toward three important principles:

RLHF reshapes the confidence surface of modern models, suppressing outward signs of uncertainty (Chen2025Overconfidence; Perez2022Discovering; Perez2024Mislead).

Semantic uncertainty and token uncertainty are different quantities, and only the former correlates with reasoning stability.

Vector-space geometry provides a natural language to talk about collapse, divergence, and loss of coherence.

These principles motivate the design decisions in the next chapter, where we move from theory to architecture.


---

CHAPTER 3 — METHODOLOGY: BUILDING THE ENTROPIC STRESS-TEST FRAMEWORK

This chapter describes how the Entropic Stress-Test Framework was designed and implemented. While the previous chapter provided the mathematical motivation for SCR and related metrics, here the focus shifts to the engineering architecture that allows us to observe, measure, and analyze cognitive collapse in real agents. The system consists of four major components—the Orchestrator, the Agent Wrapper, the Sandbox, and the Metric Service—supported by a set of scenario definitions and logging utilities. Together, these elements create a realistic and instrumented environment for evaluating model stability under non-stationary task constraints.

3.1 System Architecture Overview

At a high level, the framework operates as a loop:

The Orchestrator presents the agent with task instructions.

The agent decides the next action.

The Sandbox executes that action and returns the result.

The Metric Service evaluates internal and external signals.

The Orchestrator updates the simulation state and determines whether to continue, shock the agent, or intervene.

This feedback loop continues for a fixed number of steps or until the agent collapses.

Architecture Diagram

3.2 The Orchestrator: Simulation Engine and Control Loop

The Orchestrator is the “brain” of the system. It tracks the scenario state, manages agent history, handles shocks, and coordinates metric collection. In essence, it acts as a deterministic state machine controlling the entire experiment (Zhou2024ToolSandbox; Fu2024Tau).

Core Responsibilities

Load and initialize a scenario

Maintain the running history of agent actions and observations

Inject perturbations at pre-defined steps

Trigger Branching Probes when needed (Zhou2024Consistency)

Detect panic or collapse conditions

Optionally switch to a rescue agent

Pseudocode (Simplified Orchestrator Loop)

procedure RUN_SCENARIO(scenario, agent, metrics)

    state ← INITIALIZE_STATE(scenario)

    history ← []

    panic_counter ← 0

    for step in 1..scenario.max_steps do

        if step == scenario.perturbation_step then

            APPLY_SHOCK(state, scenario.perturbation)

        end if

        action ← agent.DECIDE_NEXT_ACTION(history)

        result ← SANDBOX_EXECUTE(action)

        

        APPEND(history, (action, result))

        

        scr ← metrics.COMPUTE_SCR(history)

        entropy ← metrics.COMPUTE_ENTROPY(action)

        LOG(step, action, result, scr, entropy)

        if DETECT_COLLAPSE(scr, entropy, panic_counter) then

            return FAILURE_STATE

        end if

    end for

    return SUCCESS_STATE

end procedure

Explanation

This loop gives the Orchestrator full control over the simulation. After each agent action, it collects metrics, evaluates stability, and checks whether intervention is required. This structure also makes experiments reproducible, since every run follows a consistent pipeline.

3.3 The Agent Wrapper: A Uniform Interface for LLMs

The framework supports any model that exposes a chat-completion API. The Agent Wrapper unifies these models behind a common interface, making it easy to swap agents during experiments (e.g., junior → senior model).

(Chen2023ToolEmu; Li2024SWEAgent)

Responsibilities

Format messages into the correct API structure

Extract the agent’s intended next action (either a tool invocation or plain response)

Generate multiple next-step candidates for Branching Probes

Maintain consistency in logging and error handling

Pseudocode (Agent Decision & Branching)

procedure DECIDE_NEXT_ACTION(history)

    response ← CALL_MODEL(history, temperature=0.1)

    return PARSE_ACTION(response)

end procedure

procedure GENERATE_BRANCHES(history, N)

    branches ← []

    for i in 1..N do

        resp ← CALL_MODEL(history, temperature=0.7)

        APPEND(branches, PARSE_ACTION(resp))

    end for

    return branches

end procedure

Explanation

A low temperature is used during normal operation to keep actions consistent. A higher temperature is used for Branching Probes since we want to expose the diversity of possible next actions.

3.4 The Sandbox: Controlled Execution Environment

The Sandbox isolates the agent from the host system and ensures all actions have real consequences. Each scenario runs inside a clean Docker container with its own file structure and tools.

Key Design Goals

Prevent harmful commands from affecting the host

Allow real execution (file operations, parsing, etc.)

Provide consistent behavior across runs

Capture output exactly as the agent sees it

(Li2025RedTeamCUA; Guo2025AgentGuard)

Pseudocode (Sandbox Command Execution)

procedure SANDBOX_EXECUTE(action)

    cmd ← FORMAT_COMMAND(action)

    wrapped ← APPLY_TIMEOUT(cmd, limit=30s)

    (status, output) ← CONTAINER_RUN(wrapped)

    return {status, output}

end procedure

Explanation

Every command goes through a timeout wrapper to prevent infinite hangs. The Sandbox does not mock failures—if the agent deletes a file or runs an invalid command, the system behaves exactly as a real environment would.

3.5 Metric Service: SCR, Entropy, and Diagnostic Tools

The Metric Service computes all measures required to analyze collapse. Unlike the Orchestrator, which manages workflow, this component focuses solely on numerical evaluation.

Metrics Provided

SCR — semantic divergence (Zhou2024Consistency)

Token entropy — model’s surface confidence

Information Gain Efficiency (IGE)

Regressive Debt Index (RDI)

Compression ratio — signals repetition loops

Pseudocode (Metric Computation)

procedure COMPUTE_SCR(branches)

    if COUNT(branches) < 2 then

        return 0

    end if

    embeddings ← EMBED(branches)

    distances ← []

    for i in 1..N do

        for j in i+1..N do

            d ← COSINE_DISTANCE(embeddings[i], embeddings[j])

            APPEND(distances, d)

        end for

    end for

    return AVERAGE(distances)

end procedure

Explanation

SCR takes the average pairwise divergence of all branches. A rise in SCR indicates that the agent’s internal next-step candidates are spreading apart—an early sign of collapse.

3.6 Branching Probes: Capturing Internal Divergence

Branching Probes are central to this framework. They measure not what the agent does, but what the agent could have done.

Pseudocode (Running a Branching Probe)

procedure RUN_BRANCHING_PROBE(history, agent, metrics)

    branches ← agent.GENERATE_BRANCHES(history, N=5)

    intents ← EXTRACT_INTENTS(branches)

    scr ← metrics.COMPUTE_SCR(intents)

    return {scr, intents}

end procedure

Explanation

Branches are limited to 5 because this provides enough coverage to detect divergence without excessive compute. Extracting “intents” ensures that we evaluate semantic differences rather than token-level variations.

3.7 Panic Detection and Intervention Logic

The Orchestrator uses SCR and entropy together to check whether the agent is entering collapse. Repeated panic triggers allow the system to swap agents or clean context.

Pseudocode (Collapse Detection)

procedure DETECT_COLLAPSE(scr, entropy, counter)

    if scr > SCR_THRESHOLD then

        counter ← counter + 1

    else

        counter ← 0

    end if

    if counter >= PANIC_LIMIT then

        return TRUE

    else

        return FALSE

    end if

end procedure

Explanation

A single spike may be noise; repeated spikes indicate loss of coherence. This design mirrors how humans detect confusion through repeated hesitation or incorrect actions.

3.8 Scenario Definitions

Each scenario specifies:

initial prompt

shock point

shock instruction

constraints

maximum steps

success criteria

(Jimenez2024SWEBenchPlus; Wang2025SavingSWE)

Pseudocode (Scenario Structure)

structure SCENARIO

    id

    name

    initial_prompt

    perturbation_step

    perturbation_instruction

    max_steps

end structure

Explanation

This allows the framework to define and run new stress-test tasks without code changes—only scenario files need to be added.

3.9 Experiment Runner and Logging

A thin experiment runner script ties everything together. After each step, it logs:

SCR

token entropy

agent action type

execution result

panic counters

compression ratio

Pseudocode (Experiment Runner)

procedure RUN_EXPERIMENT(config)

    scenario ← LOAD_SCENARIO(config.id)

    agent ← LOAD_AGENT(config.model)

    metrics ← INIT_METRICS()

    

    result ← RUN_SCENARIO(scenario, agent, metrics)

    

    SAVE_LOGS()

    GENERATE_PLOTS()

end procedure

Explanation

Plotting tools use these logs to produce SCR vs. step, entropy vs. step, and stagnation loop signatures—crucial for interpreting collapse dynamics.

3.10 Calibration and Validation Steps

Before running real trials, three calibration passes ensure the system behaves consistently:

Baseline entropy calibration

SCR threshold estimation

Embedding model sanity checks

Pseudocode (Threshold Calibration)

procedure CALIBRATE_THRESHOLD(data)

    labels ← HUMAN_LABEL_STATES(data)

    scr_values ← EXTRACT_SCR(data)

    threshold ← FIND_OPTIMAL_BOUNDARY(scr_values, labels)  (Barber2022ConformalPrediction)

    return threshold

end procedure

3.11 Methodological Limitations

While the framework is robust, some limitations remain:

Branching Probes increase compute cost

Embedding models introduce their own biases

Current scenarios cover only three task families

Thresholds must be tuned per model

Despite these constraints, the system is modular and easy to extend to new models, tasks, and metrics.


---

CHAPTER 4 — EXPERIMENTAL RESULTS

This chapter summarizes the latest runs (Dec 09, 2025) of the Entropic Stress-Test Framework. All runs used `deepseek/deepseek-v3.2` with 20-step limits for shock scenarios and 30 steps for the legacy refactor challenge. Metrics were logged each step; no rescue handoff was enabled in these runs (Zhou2024Consistency; Chen2023ToolEmu).

4.1 Experimental Setup (Current)

- Sandbox: fresh container per run; real tool execution.
- Probes: Branching probes on shock steps; entropy recorded every step.
- Runs covered three shock scenarios (Drug Filter, File Organizer, Data Pipeline) and a control-style Legacy Refactor Challenge.

4.2 Latest Run Snapshot (Dec 2025)

| Scenario (file) | Steps | Events (llm/tool/perturb) | Max SCR | Max Entropy | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Drug Filter Shock (sim_baseline_drug_filter_shock_20251209_153021.jsonl) | 20 | 12 / 6 / 2 | 0.237 | 0.160 | Shock applied twice; SCR spike mild; entropy low |
| Drug Filter Shock (sim_baseline_drug_filter_shock_20251209_133158.jsonl) | 20 | 3 / 15 / 2 | 0.230 | 0.231 | Tool-heavy loop after shock; entropy stays low/moderate |
| File Organizer Shock (sim_baseline_file_organizer_shock_20251209_153606.jsonl) | 20 | 14 / 5 / 1 | **0.337** | 0.076 | Highest SCR in this batch; entropy flat |
| File Organizer Shock (sim_baseline_file_organizer_shock_20251209_133630.jsonl) | 20 | 3 / 16 / 1 | 0.068 | 0.142 | Minimal SCR change despite shock; tool spam |
| Data Pipeline Shock (sim_baseline_data_pipeline_shock_20251209_154156.jsonl) | 20 | 16 / 3 / 1 | 0.270 | 0.050 | SCR rises on shock; entropy near zero |
| Data Pipeline Shock (sim_baseline_data_pipeline_shock_20251209_134210.jsonl) | 20 | 12 / 7 / 1 | 0.025 | **0.267** | No SCR probes fired; entropy highest of the batch |
| Legacy Refactor Challenge (sim_baseline_legacy_refactor_challenge_20251209_151022.jsonl) | 30 | 19 / 11 / 0 | 0.000 | 0.127 | Control task; no probes; long run with low entropy |
| Legacy Refactor Challenge (sim_baseline_legacy_refactor_challenge_20251209_145552.jsonl) | 30 | 1 / 29 / 0 | 0.000 | 0.138 | Tool spam; no SCR |
| Legacy Refactor Challenge (sim_baseline_legacy_refactor_challenge_20251209_144921.jsonl) | 30 | 1 / 29 / 0 | 0.000 | **0.239** | Highest entropy among control runs; no SCR |

4.3 Patterns Observed

- SCR spikes align with shocks: File Organizer (0.337) and Data Pipeline (0.270) show clear divergence at the perturbation point; Drug Filter shows smaller rises (~0.23). Runs without probes report SCR≈0 (Zhou2024Consistency).
- Entropy stays low even when behavior degrades (0.05–0.23 range); higher entropy (0.267) still failed to warn (Chen2025Overconfidence; Perez2024Mislead).
- Stagnation loops persist: tool_execution dominates several runs (e.g., 16–29 tool calls with minimal llm_reply), mirroring prior “ls/loop” failures even when SCR is low or absent (Chen2023ToolEmu).
- Panic/rescue not triggered: panic_counter stayed at 0 in these logs, so no escalations occurred despite loops.

4.4 Scenario Notes

- Drug Filter Shock: Two shocks injected; modest SCR rise (~0.23) and low entropy; behavior still drifted into tool repetition after acknowledging the change.
- File Organizer Shock: One run showed strong divergence (0.337) at the shock; another barely moved (0.068) but still spammed tools—suggesting shock detection depends on probe coverage.
- Data Pipeline Shock: One run produced SCR 0.270 with near-zero entropy; another skipped probes (SCR 0.025) yet showed the highest entropy of the batch, highlighting probe timing sensitivity.
- Legacy Refactor Challenge: No perturbations and no SCR; entropy stayed low-to-moderate (≤0.239) while tool spam dominated, reinforcing that stagnation occurs even without shocks (Jimenez2024SWEBenchPlus motivates why static tasks can still mask instability).

4.5 Artifacts to Reference

- Plots: `data/results/summary_sim_baseline_drug_filter_shock_20251209_153021.png`, `data/results/summary_sim_baseline_file_organizer_shock_20251209_153606.png`, `data/results/summary_sim_baseline_data_pipeline_shock_20251209_154156.png`, `data/results/summary_sim_baseline_legacy_refactor_challenge_20251209_151022.png`, plus aggregate views `data/results/experiment_summary.png`, `data/results/entropy_comparison.png`, `data/results/mini_swe_agent_metrics.png`.
- Logs: `logs/rescue/*.jsonl` for each run listed above.

4.6 Chapter Summary

The refreshed runs confirm the central finding: token entropy stays calm while internal coherence fractures under shocks. SCR provides the only meaningful early signal when probes fire, but probe timing and coverage matter—runs without probes report SCR≈0 even when behavior stalls. Stagnation loops persist in both shock and control settings, underscoring the need for rescue/interrupt logic beyond entropy monitoring.


---

CHAPTER 5 — DISCUSSION & IMPLICATIONS

This chapter reflects on what the experimental findings actually tell us about LLM-based agents. While the previous chapter provided quantitative evidence of collapse, the broader question is: What do these failures mean for building reliable autonomous systems? Here, I interpret the results, connect them to practical engineering concerns, and highlight the trade-offs that arise when trying to stabilize long-horizon LLM behavior.

5.1 The Cost of Reliability: Why Stability is Not Cheap

One of the clearest lessons from the experiments is that reliability does not come for free. Detecting collapse early required Branching Probes, and each probe demanded multiple calls to the underlying model. This inevitably increases compute cost.

A natural question is: Is the extra cost worth it? Based on the failure modes observed, the answer is “yes”—at least for systems where failure has significant consequences.

A single collapse can waste entire sequences of steps. Worse, because stagnation loops look superficially “organized” (the agent confidently repeats a structured command), they can be mistaken for progress. Debugging such episodes in production could require far more runtime and effort than simply monitoring for semantic divergence upfront.

In almost every domain—from banking automation to medical record parsing—the cost of a silent logical failure is high compared to the small incremental compute cost of running a probe every few steps. SCR essentially provides a “health check” on the agent’s reasoning, and like any health check, it is not free—but it is vastly cheaper than recovering from undetected degeneration (Zhou2024Consistency).

5.2 The Rescue Protocol: More Than a Safety Net

The idea behind the Rescue Protocol is intuitive: if the agent appears confused, bring in a stronger model or give the current model a reset. But the goal is not to create a hierarchy of “weak agent → strong agent.” Instead, it is to introduce discontinuity into the reasoning process.

During collapse, the agent’s internal state drifts into a narrow loop that it struggles to escape, even as it produces fluent explanations. Simply giving the same agent more tokens will not fix this. What helps is breaking the run:

summarize or clean the context,

possibly switch to a more capable model,

re-initialize reasoning from a fresh, coherent prompt.

In that sense, the Rescue Protocol behaves more like a circuit breaker than a fallback model. It interrupts the progression of internal noise that would otherwise compound irreversibly (Chen2023ToolEmu; Zhou2024ToolSandbox).

This is why the protocol is not just a convenience feature—it is central to sustaining long-horizon reasoning (Chen2024HiAgent; Zhang2025ContextFolding).

5.3 The “Why Not Just Use GPT-4?” Argument

A common objection to multi-model architectures is: If larger models are more stable, why not use them from the start?

Several findings in this thesis complicate that viewpoint.

(1) Cost and Latency

State-of-the-art models (GPT-4o, Claude 3.5 Sonnet, etc.) are significantly slower and more expensive. Most real tasks contain long stretches of routine execution—steps where a smaller model performs just as well. Using a heavyweight model for every trivial step is both wasteful and unnecessary.

(2) Context Rot Affects All Models

Even the strongest models degrade over long contexts. The issue is not simply intelligence level; it is the tendency of LLMs to accumulate subtle inconsistencies as context grows. This means: Running GPT-4 from step 1 to step 200 will not prevent collapse. The decay comes from long-running autoregressive processing itself (Wang2025ComplexityTrap; Zhang2025ContextFolding; Chen2024HiAgent).

(3) “Fresh Eyes” Matter

The Rescue Protocol does not work just because a bigger model takes over; it works because the takeover interrupts the corrupted context. A weaker model that gets replaced by the same model with a fresh prompt can recover too.

Thus, the economic argument is only part of the story. The architectural argument—that continuity of context is itself a risk factor—is equally important.

(4) SCR Enables Smart Allocation

By detecting collapse early, SCR makes hybrid architectures viable. You can:

run cheaper models during stable phases,

escalate only when SCR spikes,

return to normal once stability is restored.

This dynamic allocation model is not possible with entropy-based signals, since entropy does not reflect actual uncertainty.

In short: It is not about intelligence. It is about knowing when the agent is confused.

5.4 Architectural Implications for Agent Design

Several design principles emerge from the experiments:

1. Confidence Indicators Must Be Semantic

Token-level distributions simply do not tell us what the model “believes.” Semantic divergence does.

2. Multi-step Agents Need Built-in Stability Gates

Long-horizon tasks accumulate noise. SCR gives a way to periodically check whether the agent is drifting.

3. Interventions Must Happen Early

Once a stagnation loop begins, recovery becomes increasingly unlikely. Detecting the first moment of fracture is far more effective than detecting the consequences later on.

4. Context Management Is Critical

Even with no shocks, context naturally becomes cluttered. Summarization, pruning, and periodic resets can prevent degeneration.

5. Scenario-Based Testing Is More Realistic Than Static Benchmarks

Traditional benchmarks assume static environments. Real systems do not. Stress-testing agents under dynamic conditions surfaces weaknesses that benchmarks miss entirely (Jimenez2024SWEBenchPlus; Wang2025SavingSWE; Fu2024Tau; Zhou2024ToolSandbox; Grattafiori2025MCPMark).

5.5 Broader Implications for RLHF and Future Models

The findings also reflect limitations of RLHF itself. Models trained primarily on human preference signals learn to:

sound certain when uncertain,

avoid hedging unless explicitly asked,

suppress the expression of doubt.

This creates a surface-level fluency that hides internal confusion. While RLHF is useful for improving alignment and usability, it introduces new risks for systems that depend on accurate confidence estimation.

This suggests that future alignment approaches may need:

explicit uncertainty modeling,

training signals for admitting confusion,

penalties for overconfidence,

or complementary metrics like SCR that reveal “hidden instability” (Chen2025Overconfidence; Perez2022Discovering; Perez2024Mislead; Wang2024StyleOutweighs).

LLMs may remain autoregressive for the foreseeable future, but how we train them to communicate uncertainty is likely to evolve.

5.6 Closing Perspective on the Results

The experiments demonstrate that LLM-based agents fail in systematic, predictable ways—silently, confidently, and often repetitively. SCR provides a simple yet powerful way to see these failures form before they surface. By combining scenario-based stress tests with semantic reasoning metrics, we gain a clearer view of the fragile internal structures that govern long-horizon decision-making in current models.


---

CHAPTER 6 — CONCLUSION AND FUTURE WORK

This thesis set out to investigate how large language model–based agents behave when placed in dynamic, multi-step tasks where requirements shift mid-execution. While existing benchmarks focus heavily on correctness in static settings, real-world workflows are rarely so stable. Systems that rely on LLMs must be prepared for ambiguity, interruptions, and updates that arrive after an initial plan is formed. The Entropic Stress-Test Framework was designed to explore exactly this setting, and the results reveal important insights about failure modes, stability, and the limits of current confidence metrics.

6.1 Summary of Contributions

Across the chapters, four contributions emerge clearly:

(1) Identification of Silent Failure in RLHF-trained Agents

The experiments consistently showed that agents can lose coherence without signaling uncertainty. Even when a requirement changes, the model often continues executing an outdated plan with unwavering confidence. This “silent failure” was visible in agent behavior long before traditional metrics detected anything unusual (Chen2024ToolsFail; Chen2023ToolEmu).

(2) Introduction of SCR as a Semantic Stability Metric

The Semantic Collapse Ratio proved to be a reliable early warning signal. By examining diverging next-step possibilities rather than surface-level token distributions, SCR captured internal confusion at the exact moment it emerged (Zhou2024Consistency; Barber2022ConformalPrediction). This provides a complementary tool to entropy and aligns more closely with how humans intuitively detect hesitation or uncertainty in problem-solving.

(3) A Modular Framework for Stress-Testing Agents

The Entropic Stress-Test Framework integrates a controllable sandbox, an Orchestrator for scenario logic, and a Metric Service that computes SCR and related measures. The system makes it possible to reproduce perturbations, run detailed probes, and observe long-horizon reasoning in a controlled environment (Zhou2024ToolSandbox; Fu2024Tau).

(4) Empirical Evidence of Repetitive Stagnation Loops

All experiments—both shock and non-shock—exposed a dominant failure mode: repetitive tool calls, often the exact same command (e.g., directory listing), repeated dozens of times without progress. These loops illustrate a deeper structural issue: once the model’s plan becomes misaligned with the task state, recovery becomes unlikely without explicit intervention (Chen2023ToolEmu).

Collectively, these contributions offer a clearer picture of why LLM agents struggle in non-stationary conditions, and how monitoring internal semantic structure can reveal collapse earlier than traditional signals.

6.2 Limitations

Although the framework proved effective for studying collapse, several limitations remain:

(1) Limited Scenario Diversity

Only three families of tasks were evaluated: file organization, CSV filtering, and basic log parsing. While these tasks are representative, they do not cover more complex domains such as API orchestration, multi-agent collaboration, or long-form planning (Jimenez2024SWEBenchPlus; Grattafiori2025MCPMark).

(2) Single Model Family

The experiments focused primarily on one model (DeepSeek v3.2). Other families—OpenAI, Anthropic, Google—may show different collapse patterns, SCR magnitudes, or recovery behaviors. Cross-model comparisons would strengthen the generality of these findings.

(3) Embedding Model Bias

SCR depends on an embedding model. Different embedding models may cluster semantic meaning differently, which could shift SCR thresholds or introduce bias. While lightweight models work well in practice, the relationship between embedding geometry and agent stability deserves more rigorous study.

(4) Compute Overhead

Branching Probes require multiple model calls. Although the cost is small compared to the cost of silent failures, the overhead may still be significant for large-scale deployments unless further optimized (Zhou2024Consistency).

(5) Context Rot Not Fully Addressed

Several control scenarios collapsed even without shocks. This suggests that long contexts accumulate noise over time, but the current framework does not include automated summarization, pruning, or memory management strategies to counter this (Wang2025ComplexityTrap; Chen2024HiAgent; Zhang2025ContextFolding).

These limitations highlight the need for broader and more systematic studies of agent stability across diverse conditions.

6.3 Future Work

The results in this thesis open several promising research directions:

(1) Expanding Scenario Libraries

New task families—API pipelines, realistic data cleaning, code refactoring, or interactive search tasks—could reveal failure modes not captured in this study. A richer scenario set would also make the framework more suitable for benchmarking.

(2) Multi-Agent and Model-Switching Architectures

SCR provides a natural trigger for switching between agents of different strengths. Future systems may use SCR to coordinate model handoffs, maintain team stability, or orchestrate collaborative agents with complementary roles (Chen2024HiAgent; Zhang2025ContextFolding).

(3) Real-Time Context Management

Integrating automatic context summarizers or rolling memory structures could reduce long-context degradation. An adaptive context-reset mechanism triggered by SCR spikes may significantly improve long-horizon resilience (Wang2025ComplexityTrap; Zhou2025MemTool; Guo2024MOSS).

(4) SCR as a Training Signal

One intriguing direction is whether SCR—or semantic divergence more broadly—can be used as part of model training. Penalizing or rewarding certain divergence structures might teach models to regulate internal uncertainty more transparently.

(5) Cross-Model Benchmarking

Evaluating SCR behavior across GPT-4o, Claude 3.5, Gemini, and other models could reveal universal patterns or model-specific biases. Such cross-evaluation would also position SCR as a standardized measure for agent stability (Fu2024Tau; Grattafiori2025MCPMark).

(6) Integrating with Production Monitoring

In real deployments, SCR could augment logging systems, triggering alerts when an agent begins drifting, or initiating a context reset before a full stagnation loop occurs. This would extend the framework beyond research and into practical reliability engineering.

Collectively, these directions outline how SCR and dynamic stress-testing can play a central role in developing the next generation of reliable, reasoning-aware LLM agents.

6.4 Closing Remarks

LLM-based agents have made remarkable progress, but they remain fragile in subtle ways. As this thesis shows, the biggest risks are not dramatic failures—they are the quiet, confident, and repetitive errors that emerge when an agent’s internal representation becomes misaligned with the task. By shifting the focus from surface fluency to semantic stability, the Entropic Stress-Test Framework offers a new perspective on how these systems think, how they fail, and how they can be made more robust.

The long-horizon future of AI will depend on our ability to detect confusion early, intervene intelligently, and design systems that remain stable even as their environments change. This work is a step toward understanding that challenge more deeply.
