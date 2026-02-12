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
