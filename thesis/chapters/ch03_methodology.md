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
