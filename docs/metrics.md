# Metrics Definitions

## 1. Information Gain Efficiency (IGE)
Measures the reduction in entropy per unit of cost (token usage) after a tool execution.
Formula: `IGE = (H_pre - H_post) / Cost`
- **H_pre**: Entropy of the agent's thought *before* using the tool.
- **H_post**: Entropy of the agent's thought *after* receiving the tool output.
- **Cost**: Number of tokens generated/consumed.

## 2. Semantic Collapse Ratio (SCR)
Measures the diversity of the agent's potential future thoughts.
Formula: `SCR = Avg(CosineDistance(Embedding(Branch_i), Embedding(Branch_j)))`
- Higher SCR means the agent is considering diverse possibilities (High Confusion or High Creativity).
- Lower SCR means the agent has collapsed to a single line of reasoning (Confidence or Dogmatism).

## 3. Entropy (H)
Measures the uncertainty in the agent's token prediction distribution.
Calculated via `logprobs` provided by the LLM.

## 4. Compression Ratio (CR)
Measures the repetitiveness of the agent's output using `zlib`.
`CR = Compressed_Size / Original_Size`
- < 0.2 indicates looping/repetitive text.

## 5. Regressive Debt Index (RDI)
(Experimental) Measures how much the agent's current plan deviates from a known "good" trajectory or ground truth.
Formula: `RDI = CosineDistance(Current_Plan_Embedding, Ground_Truth_Embedding)`

