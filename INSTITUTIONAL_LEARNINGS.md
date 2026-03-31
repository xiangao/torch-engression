# Institutional Learnings: GPU-Accelerated Econometrics

**Date**: 2026-03-29
**Scope**: Relevant past solutions for torch-engression in context of endid, divR, dma, panel data transformations, bootstrap inference

---

## Executive Summary

Your torch-engression package is foundational for a pipeline of distributional econometrics tools. Institutional knowledge from related projects reveals **critical patterns for GPU-accelerated inference, bootstrap on GPU, panel data transformation, and weight propagation in cascaded neural network estimation**. This document consolidates learnings that prevent repeated mistakes and establish patterns to follow.

---

## Part 1: Panel Data Transformations & Staggered Adoption

### Context
Your next step after torch-engression is **endid** (lwdidR + engression for distributional DiD). This requires understanding panel data transformations applied before entering the neural network.

### Learning 1: Panel Residualization via Unit-Specific Pre-Treatment Data
**Source**: `lwdidR` (Lee & Wooldridge 2025) design and implementation
**File**: `/home/xao/projects/software/lwdidR/README.md`

**Key Insight**: The panel→cross-section transformation is the critical first step:

- **Transform step**: For each unit i, residualize outcome using **only that unit's own pre-treatment observations** (not pooled or global statistics)
- **Rolling methods**: `demean` (subtract unit mean), `detrend` (subtract linear trend), `demeanq` (demean by quintile), `detrendq` (detrend by quintile)
- **Output**: Single cross-sectional outcome `ydot_postavg` per unit (post-period average minus pre-treatment residual)
- **Critical**: This avoids selection bias from pooled residualization and preserves parallel trends assumptions

**Application to endid/torch-engression**:
- lwdidR provides `apply_transform(data, y, ivar, tvar, post, rolling, tpost1, season_var)` — to be exported in v0.1.1
- endid calls this function to get cross-section, then pipes into `engression(X = cbind(D, controls), Y = ydot_postavg)`
- **No further data manipulation needed** — engression learns nonlinear relationships directly (unlike OLS, which needs D*X interactions)

**Reference**:
- Design: `/home/xao/projects/claude/frengression/docs/superpowers/specs/2026-03-26-endid-design.md` (lines 122-142)
- Plan: `/home/xao/projects/claude/frengression/docs/superpowers/plans/2026-03-26-endid.md`

---

### Learning 2: Staggered Adoption Aggregation
**Source**: lwdidR + endid design
**Files**: lwdid README section "Algorithm Notes" + endid design lines 134-142

**Key Insight**: For staggered adoption designs, **there are multiple correct aggregation strategies**:

**Option 1: Pooled cross-sectional (recommended for endid)**
- Treated units: use own-cohort `ydot_postavg`
- Never-treated units: weighted average across cohorts (w_g = n_treated_g / n_total_treated)
- Fit single engression model on pooled N = N_treated + N_never_treated
- Yields single overall ATT + single QTE curve

**Option 2: Per-cohort engression (endid v0.1)**
- For each cohort g: build cross-section (cohort g units + never-treated), fit engression
- Get cohort-specific ATT and QTE
- Optionally aggregate via simple average or other weighting

**Option 3: Out of scope for v0.1**
- (g,r)-specific engression: sample sizes too small at individual (cohort × period) level for neural network

**Gotcha**: Overall ATT from pooled engression ≠ simple average of per-cohort ATTs
**Why**: Per-cohort models use different control distributions; pooled model uses single common control distribution.

---

## Part 2: Bootstrap Inference on GPU

### Learning 3: Unit-Level Bootstrap for Engression SEs
**Source**: endid design + dma/divR precedent
**Files**: endid design lines 144-154; dma CLAUDE.md; torch-engression README (AMP precision)

**Key Insight**: Engression has **no analytical SEs** (neural network trained via stochastic optimization). Solution: **unit-level bootstrap**.

**Algorithm**:
```
for b in 1:nboot {
  1. Resample units WITH REPLACEMENT from cross-section
  2. Refit engression on resampled data
  3. Compute ATT (trimmed mean difference) and QTE (quantile contrasts)
  4. Store estimates
}
SE = sd(bootstrap_estimates)
CI = percentile interval [alpha/2, 1-alpha/2]
```

**Implementation Notes** (from torch-engression patterns):
- **For staggered designs**: Bootstrap within cohorts proportionally (preserve cohort weights)
- **Default nboot = 200** (configurable) — balance between accuracy and runtime
- **GPU acceleration**: If bootstrap refits happen on GPU, get 15x speedup per refit (as shown in torch-engression benchmark)
- **Reproducibility**: Set seed for unit-level resampling in each bootstrap iteration if needed

**Critical**: In endid design, bootstrap is done **on the cross-section level** (N ≈ 50-100 units), not on original panel (N_obs ≈ 1000s). Much faster than raw panel bootstrap.

---

### Learning 4: AMP Training Precision (Categorical Stability)
**Source**: torch-engression design
**File**: `/home/xao/projects/claude/torch-engression/README.md` and CLAUDE.md

**Key Insight**: When combining engression with bootstrap on GPU, **be careful with loss computation precision**:

- **Forward passes**: FP16 (fast) ✓
- **Loss computation**: FP32 (stable) ← **CRITICAL**
- **Reason**: Energy loss involves $s_1 - s_2/2$ (catastrophic cancellation in FP16)

**Pattern from torch-engression**:
```python
with autocast(dtype=torch.float16):
    # Forward pass in FP16
    z = model(x)  # fast matrix multiplies
# Loss in FP32
loss = energy_loss(z, y)  # numerically stable
```

**Application**: When refitting engression in bootstrap iterations, use this same AMP scope. torch-engression's `train()` method already implements this.

---

## Part 3: Distributional Causal Inference Pipeline

### Learning 5: Engression is Foundation for Distributional Methods
**Source**: frengression CLAUDE.md + dma + divR architecture
**File**: `/home/xao/projects/claude/frengression/CLAUDE.md`

**Key Insight**: Your torch-engression package enables three related distributional econometrics methods:

| Method | Upstream | Downstream | Next Step |
|--------|----------|-----------|-----------|
| **torch-engression** | ✓ Complete | — | Neural network distributional regression |
| **endid** | lwdidR + torch-engression | — | Distributional DiD (staggered + bootstrap) |
| **dma** | engression + S7 data structures | Riesz NNs | Distributional mediation (4 effect types) |
| **divR** | torch-engression | — | Distributional IV (unmeasured confounding) |

**Critical dependency order**:
1. torch-engression (base distributional regression)
2. endid (panel transformations + distributional DiD)
3. torch-frengression (3 StoNets for conditional, marginal, residual distributions)
4. dma/divR (specialized causal structures built on top)

---

### Learning 6: Observation Weights Must Propagate Through All Stages
**Source**: dma (Distributional Mediation Analysis)
**File**: `/home/xao/projects/software/dma/CLAUDE.md`

**Key Insight**: When implementing weighted estimation (e.g., IPW, causal forests), **observation weights must propagate through cascaded neural networks**, not just at the final EIF stage.

**Pattern from dma**:
```
Outcome regressions (theta):
  - Nested cascade: theta3 → theta2 → theta1
  - Each level uses engression() with obs weights w
  - NOT just final level

Density ratio regressions (alpha):
  - Riesz representer NNs at each cascade level
  - Loss: E[alpha²] - 2*w*f(alpha, X')  ← weights in loss
  - NOT post-hoc reweighting
```

**Why**: Ignoring weights in intermediate stages biases nuisance estimates, inflating variance of final treatment effect.

**Gotcha**: `prepare_engression_x()` drops zero-variance columns. When predicting on shifted/intervention data, **always pass `ref_cols` from training data** to prevent column dropping that breaks alignment.

---

### Learning 7: Riesz Representer NNs for Density Ratios
**Source**: dma implementation
**File**: `/home/xao/projects/software/dma/R/riesz_nn.R` + CLAUDE.md

**Key Insight**: Density ratio estimation $p(y|intervention)/p(y|observed)$ can be done via neural networks trained to minimize:
$$\text{Loss} = \mathbb{E}[\alpha^2] - 2\mathbb{E}[w \cdot f(\alpha, X')]$$

where $f$ is output of Riesz NN and $w$ are observation weights.

**When you'll need this**:
- endid + indirect effects (mediation via distributional channel)
- Weighted counterfactual density sampling
- Importance weighting via Riesz representations

**Current status**: Not in torch-engression yet (focus is univariate neural nets for regression). dma implements this in R + torch.

---

## Part 4: GPU Acceleration Patterns

### Learning 8: Device Auto-Detection & MPS Gotchas
**Source**: torch-engression design
**File**: `/home/xao/projects/claude/torch-engression/README.md` + `src/torch_engression/utils.py`

**Key Insight**: Auto-detect device in preference order: **CUDA > MPS > CPU**

**Pattern**:
```python
def auto_device(device=None):
    if device is None:
        return (torch.device("cuda") if torch.cuda.is_available()
                else (torch.device("mps") if torch.backends.mps.is_available()
                else torch.device("cpu")))
```

**MPS (Apple Silicon) Warning**: `torch.cdist` has known issues on MPS. torch-engression auto-detects and issues warning.

**Application to bootstrap**: Each refitting loop on bootstrap iteration b:
- Model stays on same device
- Data moved to device once
- No repeated device transfers (slow)

---

### Learning 9: Chunked Sampling Avoids OOM
**Source**: torch-engression design
**File**: `/home/xao/projects/claude/torch-engression/README.md` (line 29)

**Key Insight**: When sampling from trained engression (predict with sample_size=500), generate in chunks to avoid OOM:

```python
# Bad: sample_size=500, batch_size=100 → (batch, d_y, 500) OOM on large batches
samples = model.sample(X, sample_size=500)

# Good: default chunk_size=50 keeps memory flat
# Internally: loop over chunks, concatenate
```

**Application to endid bootstrap**: When computing counterfactual densities for QTE (n units × sample_size=500), use chunked sampling.

---

## Part 5: Critical Patterns & Gotchas

### Pattern 1: Standardization Must Be Reversible
**Source**: torch-engression + dma + divR
**Files**: torch-engression CLAUDE.md line 26; dma + divR both use standardize=True

**Gotcha**: Engression standardizes inputs internally. When reporting counterfactual predictions, **de-standardize using original mean/sd**.

**Pattern**:
```python
# Training
engression(X_std, Y_std, standardize=True)  # stores mean_X, sd_X, mean_Y, sd_Y

# Prediction + de-standardization
y_pred_std = model.predict(X_test_std, target="mean")
y_pred_original_scale = y_pred_std * sd_Y + mean_Y
```

---

### Pattern 2: Nonlinear Models Need Different Control Specifications
**Source**: endid design (lines 129-130)

**Gotcha**: In lwdidR (OLS), control variables are entered **raw + interaction with D**. In endid (engression):
- Control variables: **raw values only**
- No D*X interactions needed
- Neural network learns nonlinear relationships directly

**Why**: OLS is linear; interactions explicit. Engression is nonlinear; interactions learned implicitly.

---

### Pattern 3: Parameter Names Must Match Original
**Source**: torch-engression design
**File**: README.md (line 28)

**Pattern**: When porting algorithms to PyTorch, **preserve parameter names from original Python/R implementation**.

Example from torch-engression:
- `num_layer` not `num_layers` (engression convention)
- `hidden_dim` not `hidden_size`
- `noise_dim` not `noise_dimension`

**Why**: Users familiar with engression can switch to torch-engression without relearning API. Reduces migration friction.

---

### Pattern 4: torch.compile is Opt-In with Warmup Cost
**Source**: torch-engression design
**File**: README.md lines 74-79

**Gotcha**: `torch.compile` can improve performance on long training runs but adds compilation overhead upfront.

**Pattern**:
- Short training runs (< 50 epochs): `compile_model=False` (default)
- Long training runs (> 500 epochs): Consider `compile_model=True`
- Dynamic features (noise injection, chunked sampling): Works with dynamo but adds warmup

---

## Part 6: Cross-Project Integration Points

### Integration 1: lwdidR → endid → torch-frengression
**Status**: lwdidR complete, endid in planning phase (Task 0: export apply_transform)

**Next step**: Export one function from lwdidR:
```bash
# In ~/projects/software/lwdidR/R/transform.R line 19
# Change: @keywords internal
# To: @export

# Bump DESCRIPTION version to 0.1.1
# Regenerate NAMESPACE via roxygen2::roxygenise()
```

See plan: `/home/xao/projects/claude/frengression/docs/superpowers/plans/2026-03-26-endid.md` (Tasks 0-4)

---

### Integration 2: engression → torch-engression Parity
**Status**: torch-engression complete (15.1x GPU speedup, 45 tests pass)

**Pattern**: torch-engression and R/Python engression should produce **identical predictions** on same data:
- Same seed
- Same hyperparameters
- Monte Carlo error < 0.01 SD from FP32 precision

Verified in benchmark: `/home/xao/projects/claude/torch-engression/nb/benchmark.ipynb`

---

### Integration 3: Weights in Riesz NNs (dma pattern)
**Status**: Implemented in dma R package; pattern ready for torch-engression extensions

**When needed**: If you extend torch-engression to support observation weights in training:
```python
# Current: unweighted
loss = energy_loss_two_sample(pred_1, pred_0)

# Future pattern (from dma):
# For Riesz NNs, loss must include weights in the gradient
loss = (alpha ** 2).mean() - 2 * (weights * f_alpha).mean()
```

---

## Part 7: Testing & Validation Patterns

### Pattern 5: Consistency Tests Against Simpler Alternatives
**Source**: endid design (lines 197-204)

**Test strategy**:
1. Generate simple DGP with **constant treatment effect**
2. Fit endid, lwdidR side-by-side
3. Verify `endid_ATT ≈ lwdidR_ATT` within tolerance (e.g., 5% SD)
4. **Why**: Engression should collapse to linear when truth is linear; identifies neural network misspecification

---

### Pattern 6: Distributional Tests Detect Heterogeneity
**Source**: endid design (lines 200-202)

**Test strategy**:
1. Generate DGP where **treatment effect ∝ Y quantile** (e.g., effect = 2 * quantile)
2. Fit endid
3. Verify QTE is monotonically increasing (low quantile small effect, high quantile large effect)
4. **Why**: Tests that distributional methods detect heterogeneity; not just point averages

---

## Part 8: Next Steps for torch-engression Ecosystem

### Immediate (Week 1-2): Prepare endid
- [ ] Export `apply_transform()` from lwdidR v0.1.1
- [ ] Create skeleton endid package at `/home/xao/projects/software/endid/`
- [ ] Implement common-timing path + tests
- [ ] Implement staggered path + tests

### Near-term (Week 3-4): endid + Bootstrap
- [ ] Bootstrap inference for ATT, QTE (unit-level resampling)
- [ ] S3 methods: predict (att, qte, cate, counterfactual), plot (qte curve, density)
- [ ] Castle Doctrine vignette (replicate lwdidR results + add distributional outputs)

### Medium-term (Month 2): torch-frengression
- [ ] Three StoNet architecture (conditional, marginal, residual)
- [ ] Two-stage training (outcome first, then treatment/confounder marginal)
- [ ] Counterfactual sampling for causal inference benchmarking

### Long-term (Month 3+): Extended methods
- [ ] torch-dma: Distributional mediation with weighted Riesz NNs
- [ ] torch-divR: Distributional IV with two coupled StoNets
- [ ] Unified inference framework (bootstrap + influence functions)

---

## Summary: Do's and Don'ts

### DO:
- Use unit-level bootstrap for engression SEs (no analytical SEs available)
- Keep AMP loss computation in FP32 (catastrophic cancellation risk in FP16)
- Apply panel transformations (unit-specific residualization) before entering engression
- Propagate observation weights through all cascade stages (not just final)
- Preserve original parameter names (num_layer, noise_dim, hidden_dim)
- Use chunked sampling to avoid OOM on large predictions
- Test distributional methods against constant-effect DGPs (consistency check)

### DON'T:
- Use analytical SEs from engression (they don't exist; use bootstrap)
- Forget to de-standardize predictions (engression standardizes internally)
- Add D*X interactions to engression inputs (learn nonlinearity implicitly)
- Use global residualization (use unit-specific pre-treatment residuals in panel data)
- Ignore observation weights in intermediate neural network stages
- Pass shifted data to `prepare_engression_x()` without `ref_cols` (drops constant columns)
- Use torch.compile on short training runs (warmup overhead dominates)

---

## Files to Reference

| Topic | File |
|-------|------|
| Panel transformations | `/home/xao/projects/software/lwdidR/README.md` |
| endid design | `/home/xao/projects/claude/frengression/docs/superpowers/specs/2026-03-26-endid-design.md` |
| endid plan | `/home/xao/projects/claude/frengression/docs/superpowers/plans/2026-03-26-endid.md` |
| torch-engression | `/home/xao/projects/claude/torch-engression/README.md` + CLAUDE.md |
| dma (weight propagation) | `/home/xao/projects/software/dma/CLAUDE.md` |
| divR (energy loss) | `/home/xao/projects/software/divR/CLAUDE.md` |
| frengression (architecture) | `/home/xao/projects/claude/frengression/CLAUDE.md` |

---

**End of Institutional Learnings**
