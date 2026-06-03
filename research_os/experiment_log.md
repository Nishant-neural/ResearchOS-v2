# Persistent Neural Memory — Current Experimental Context Summary

## Core Architecture

Research direction:

Persistent neural memory using cached encoder hidden states instead of traditional text-based RAG.

Pipeline:

```text
offline encoding
→ hidden-state storage
→ latent retrieval
→ decoder conditioning
```

Model:

```text
FLAN-T5-base
```

Storage:

```text
Qdrant
```

Memory representation:

```text
final encoder hidden states
```

Generation method:

```text
decoder cross-attention over concatenated latent memories
```

---

# Major Verified Findings

## 1. Hidden-State Reuse Works

The central hypothesis is now strongly supported mechanically.

Verified:

* encoder hidden states can be serialized
* stored externally
* retrieved later
* concatenated dynamically
* directly reused for generation

without rerunning the encoder.

The decoder can generate coherent outputs from retrieved latent memories.

This alone establishes that encoder hidden states can function as reusable semantic memory artifacts.

---

# 2. Final Encoder Layer Is Special

Controlled layer experiments showed:

```text
Layer 6  -> repetitive degeneration
Layer 11 -> repetitive degeneration
Layer 12 -> coherent generation
```

Verification:

```text
hidden_states[12] == last_hidden_state
```

Mean difference:

```text
0
```

Interpretation:

Intermediate layers contain information but are not decoder-readable directly.

The final encoder block appears to produce a specialized decoder-interface representation.

Key insight:

```text
information-bearing
≠
decoder-consumable
```

Current hypothesis:

The final encoder layer acts as a latent interface protocol between encoder and decoder.

---

# 3. Earlier Experiments Were Affected By Major Chunking Bug

Important bug discovered:

Chunking was accidentally word-based instead of tokenizer-based.

Expected:

```text
512-token chunks
```

Actual:

```text
1000–1500+ token chunks
```

Often spanning:

```text
3–4 PDF pages
```

Consequences:

Single chunks contained:

* formulas
* references
* methods
* results
* multiple concepts
* unrelated semantic regions

This created severe semantic contamination.

---

# 4. Reinterpretation Of Earlier Results

Initially many failures were attributed to hidden-state limitations.

Current evidence suggests most degradation came from:

```text
oversized chunks
+
semantic contamination
+
attention dilution
```

NOT from hidden-state reuse itself.

Important observation:

Even with extremely oversized latent memories:

outputs remained:

* grammatical
* coherent
* semantically related

Failures were usually:

* semantic drift
* nearby concept substitution
* generalized summarization

NOT decoder collapse.

This is critical evidence that latent memory itself remains usable under heavy load.

---

# 5. Attention Allocation Appears To Be The Main Bottleneck

Current evidence suggests:

Storage bottleneck:

```text
No evidence.
```

Latent-memory corruption bottleneck:

```text
No evidence.
```

Decoder compatibility bottleneck:

```text
Solved for final encoder layer.
```

Main bottleneck now appears to be:

```text
decoder attention allocation
```

Key observation:

```text
more latent memory
≠
nonsense

more latent memory
→
semantic drift
```

This resembles long-context attention degradation in transformers.

---

# 6. README Experiments Showed Stable Semantic Behavior

README collection:

```text
12 chunks total
```

Experiments:

```text
top_k = 2 → stable answer
top_k = 5 → stable answer
top_k = 10 → stable answer
top_k = 12 → semantic abstraction drift
```

Observed final answer at full-memory load:

```text
Latent semantic memory is a semantic computation artifact that can be reused and reused.
```

Interpretation:

As latent memory size increased, the decoder transitioned from:

```text
query-conditioned retrieval
```

toward:

```text
global semantic averaging
```

This suggests a latent-memory phase transition.

---

# 7. Synthetic Semantic Interference Corpus Introduced New Failure Modes

A synthetic multi-domain interference corpus was created containing:

* astrophysics
* psychology
* economics
* Roman engineering
* quantum computing
* music theory
* distributed systems
* etc.

Purpose:

Controlled semantic interference testing.

This corpus produced fundamentally different behavior from research PDFs.

---

# 8. Discovery Of Semantic Superposition Effects

Example experiment:

Question:

```text
How does neutron star form?
```

Results:

```text
2 chunks → "fusion"
```

instead of:

```text
after supernova explosions
```

Interpretation:

The decoder retrieved the correct semantic domain:

```text
astrophysics
```

but failed to preserve exact relational bindings.

This suggests latent memories preserve:

```text
semantic fields
```

better than:

```text
precise symbolic relations
```

---

# 9. Discovery Of Beam Search Collapse

Important observation:

With 3–4 semantically distant chunks:

beam search sometimes generated:

```text
"."
```

Debugging showed:

* hidden states numerically healthy
* attention masks correct
* no NaNs
* no tensor corruption

Debug statistics:

```text
hidden_states.std() ≈ 0.14
max abs value ≈ 0.9
```

This proved the failure was NOT caused by implementation bugs.

---

# 10. Sampling Revealed Decoder Was Still Functional

When beam search was disabled:

same latent memory produced coherent outputs.

Example:

```text
a system of atoms interacting with each other
```

Interpretation:

The decoder retained semantic signal.

Beam search was collapsing under:

```text
high semantic uncertainty
```

rather than latent corruption.

Key insight:

Latent memory retrieval appears to produce:

```text
distributed semantic activation
```

rather than explicit token-level evidence.

Beam search handles this poorly because probability distributions become flatter and more ambiguous.

---

# 11. Emerging Theory — Latent Semantic Superposition

Current evidence increasingly supports:

```text
decoder cross-attention over latent memories
=
semantic field composition
```

rather than traditional retrieval.

Observed phenomena now fit one unified framework:

| Observation            | Interpretation                           |
| ---------------------- | ---------------------------------------- |
| semantic drift         | blended semantic attractors              |
| global summarization   | semantic averaging                       |
| beam collapse          | uncertainty collapse                     |
| coherent weird outputs | latent interpolation                     |
| nearby wrong answers   | semantic nearest-neighbor reconstruction |

---

# 12. Most Important Current Hypothesis

Persistent neural memory is likely feasible.

The primary limitation is probably NOT:

```text
memory storage
```

but:

```text
decoder interpretation capacity
```

Specifically:

* attention competition
* semantic interference
* latent manifold compositionality
* decoding instability under semantic superposition

---

# Current Most Important Open Questions

## A. Is Semantic Distance The Real Bottleneck?

Hypothesis:

Same-domain latent memories compose stably.

Cross-domain latent memories destabilize decoder interpretation.

Planned experiment:

```text
same-domain chunks
vs
related-domain chunks
vs
maximally distant chunks
```

Measure:

* exact correctness
* semantic drift
* abstraction frequency
* beam collapse frequency

---

# B. Does Cross-Attention Function As A Memory Reader?

Emerging hypothesis:

```text
cross-attention
=
latent memory retrieval mechanism
```

with finite semantic precision.

---

# C. Is Latent Memory Fundamentally More Semantic Than Symbolic?

Current evidence suggests latent memory may naturally encode:

```text
distributed semantic manifolds
```

rather than exact extractive facts.

This may explain:

* semantic smoothing
* concept blending
* generalized abstraction

during generation.

---

# Current Recommended Experimental Priorities

1. tokenizer-correct chunking only
2. chunk_size = 150–256
3. top_k scaling experiments
4. same-domain vs cross-domain composition tests
5. beam vs sampling comparisons
6. semantic drift categorization
7. latent manifold compatibility studies

---

# Current Research State

The project has progressed far beyond:

```text
"Can hidden states work at all?"
```

Current research question is now closer to:

```text
"How do decoder cross-attention dynamics behave when externally persisted latent semantic memories are composed together?"
```

This has become a study of:

* latent memory composition
* semantic superposition
* attention allocation
* manifold compatibility
* decoder uncertainty dynamics

rather than merely hidden-state reuse.
