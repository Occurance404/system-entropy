# Reproducibility Commands (Supplement)

These commands regenerate the primary SWE-PolyBench campaign closeout artifacts used in the paper.

```bash
cd ~/DoNootTouch/ai-problem

./.venv/bin/python analysis/closeout_swepolybench_campaign.py \
  --run-tag 20260225_231718 \
  --models-json benchmarks/models.openrouter.remaining5.json

./.venv/bin/python analysis/failure_forensics_swepolybench.py \
  --run-tag 20260225_231718

./.venv/bin/python analysis/build_results_appendix.py \
  --run-tag 20260225_231718 \
  --two-model-tag 20260222_123759
```

GLM-5 follow-up closeout:

```bash
./.venv/bin/python analysis/closeout_swepolybench_campaign.py \
  --run-tag 20260305_020845_glm5 \
  --models-json benchmarks/models.openrouter.glm5_only.json

./.venv/bin/python analysis/failure_forensics_swepolybench.py \
  --run-tag 20260305_020845_glm5
```
