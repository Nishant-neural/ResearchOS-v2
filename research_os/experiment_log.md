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


Experiment Log — Latent Superposition Test

Initial latent composition used mean pooling over retrieved hidden states. This caused severe latent collapse, producing trivial outputs such as "a" regardless of retrieved document content. Hidden-state statistics showed extremely low activation variance (std ≈ 0.06), suggesting semantic information destruction.

The architecture was then changed from mean pooling to additive latent superposition:

(256,768) + (256,768) -> (256,768)

Instead of concatenation or averaging, retrieved hidden states were directly added together.

Unexpectedly, despite major implementation flaws:

variable-length latent tensors
truncation to minimum sequence length
accidental addition of query/instruction hidden states
no semantic token alignment

the model still produced semantically related outputs.

Example query:

What are galaxies?

Generated outputs:

planetary
astronomes

Although incorrect, the outputs remained inside the astronomy semantic neighborhood rather than collapsing into meaningless tokens.

Latent statistics also improved substantially:

activation std increased from ~0.06 to ~0.086
average token norm increased to ~2.38

This suggests additive superposition preserved significantly more semantic structure than mean pooling.

Most importantly:

Even under severe architectural mistakes, latent addition still retained coarse semantic category information.

This is evidence that transformer hidden states may support meaningful semantic field composition rather than behaving purely as symbolic memory representations.

Experiment Log — Latent Superposition vs Decoder Stability

Early latent superposition experiments unexpectedly produced more semantically meaningful outputs despite severe architectural flaws.

The earlier implementation used:

variable-length encoder hidden states
truncation to minimum overlapping sequence length
direct addition of query + memory + instruction hidden states
no proper masking
partial token overlap only

Example flawed operation:

min_len = min(
    base_memory.shape[1],
    memory.shape[1],
)

base_memory[:, :min_len, :] += (
    memory[:, :min_len, :]
)

Despite being mathematically incorrect, outputs remained semantically related to the target domain.

Example:

Question:

What are galaxies?

Generated outputs:

planetary
astronomes

These outputs were incorrect but remained strongly astronomy-adjacent.

After architectural “fixes” were introduced:

fixed-length padding
full latent superposition
proper attention masks
larger latent sequence composition
stronger memory accumulation

generation quality unexpectedly became worse.

Outputs collapsed into:

[
a)
victim

or other unstable malformed completions.

This suggests a major theoretical insight:

semantic persistence and decoder compatibility are different properties

The earlier broken system accidentally preserved decoder stability because only a small portion of the latent manifold was perturbed.

The truncation bug effectively acted as:

small residual latent conditioning

rather than full semantic superposition.

As a result:

most encoder hidden states remained transformer-native
positional structure remained partially intact
decoder cross-attention stayed inside familiar activation manifolds

The newer mathematically cleaner system produced larger latent perturbations that violated transformer-native geometry.

This suggests:

transformer hidden states may not behave as linearly composable semantic vectors

at least not under naive tokenwise addition.

More specifically:

semantic information can survive latent algebra
decoder coherence is much more fragile
transformers appear tolerant to small latent perturbations
large synthetic manifold violations destabilize generation

Important observation:

The earlier flawed system preserved semantic category information better than the theoretically cleaner implementation.

This may indicate that:

small residual semantic modulation

is more compatible with transformer decoding than:

full hidden-state superposition

Current hypothesis:

Transformer decoders require encoder hidden states to remain close to naturally generated manifold trajectories. Large algebraic manipulations may preserve semantic signal while destroying decoder interpretability.

Experimental implication:

Future work should likely explore:

residual latent steering
low-rank semantic perturbations
attention-based memory conditioning
learned latent composition
manifold-preserving hidden-state operations

instead of unrestricted tokenwise hidden-state addition.

Experiment Log — Latent Superposition, Re-Ingestion, and Semantic Field Preservation
Objective

Evaluate whether encoder hidden states can function as persistent neural memories and whether multiple retrieved hidden states can be composed through latent superposition rather than text concatenation.

Initial Mean-Pooling Experiment

Retrieved hidden states were combined using mean pooling.

Result:

a

or similarly collapsed outputs regardless of retrieved content.

Observed statistics:

std ≈ 0.06

Interpretation:

Mean pooling destroyed most latent structure and produced severe semantic collapse.

First Latent Addition Experiment

Mean pooling was replaced with direct hidden-state addition.

Implementation was later discovered to contain major flaws:

variable-length hidden states
truncation to minimum sequence length
query hidden states merged with memory hidden states
instruction hidden states merged with memory hidden states
inconsistent attention masks

Example operation:

base_memory[:, :min_len, :] += memory[:, :min_len, :]

Despite these flaws, outputs became:

planetary
astronomes

for:

What are galaxies?

Important observation:

The outputs were incorrect but remained inside the astronomy semantic neighborhood.

Architectural Cleanup

The system was then modified to:

re-ingest all documents using fixed latent width
use fixed-length latent memories
separate query framing from memory storage
separate answer instructions from memory storage
preserve full latent sequence lengths
construct proper attention masks

After re-ingestion, latent statistics became:

std ≈ 0.147
avg token norm ≈ 4.0

indicating substantially stronger latent activations.

Control Experiment: No Retrieved Memory

Question:

What are galaxies?

Retrieved memories:

none

Outputs:

a group of objects with a similar shape
a constellation
a space station

Observed statistics:

std ≈ 0.058
avg token norm ≈ 0.64

Interpretation:

The model possessed weak astronomy-related prior knowledge but failed to produce a correct definition.

Generations remained generic and highly uncertain.

Experiment: Retrieved Hidden Memories Enabled

Question:

What are galaxies?

Retrieved memories included astrophysics chunks containing explicit galaxy definitions:

"Galaxies are gravitationally bound systems containing stars, gas, dust, dark matter..."

Generated outputs:

The scientific name for a planet is.

and

A series of planets orbiting stars, and a variety of other planets in the Solar System.

These answers remained incorrect.

However they were:

grammatically coherent
astronomy-domain grounded
significantly more detailed than the no-memory baseline
Most Important Observation

Comparison:

Without Memory
constellation
space station
group of objects
With Memory
planet
stars
solar system
orbiting

The semantic center of generation clearly shifted.

The decoder consistently moved deeper into the astronomy domain after memory injection.

Key Insight

The retrieved memory contained an explicit galaxy definition, yet the model never reconstructed that definition.

Instead the decoder produced nearby astronomy concepts.

This suggests:

galaxy memory
→ astronomy semantic field
→ decoder output

rather than:

galaxy memory
→ galaxy definition reconstruction
Updated Hypothesis

Current evidence suggests latent superposition preserves:

high-level semantic category information

while degrading:

fine-grained conceptual information

The decoder appears able to identify:

"This latent state is about astronomy."

but not:

"This latent state specifically encodes galaxies."
Theoretical Implication

The experiments now support a stronger claim than earlier runs.

Evidence currently suggests:

latent hidden states contain recoverable semantic fields

and

latent superposition can preserve domain-level meaning

even when exact factual reconstruction fails.

This is consistent with the emerging hypothesis that transformer hidden states may behave more like:

continuous semantic fields

rather than:

symbolic memory records
Current Working Conclusion

Observed behavior is best summarized as:

hidden-state superposition
→ semantic steering survives

hidden-state superposition
→ domain identity survives

hidden-state superposition
→ exact concept reconstruction degrades

This is presently the strongest empirical evidence obtained in support of the semantic-field interpretation of latent memory representations.


Updated Interpretation After Statistical Validation and Proposition Retrieval Experiments
Major Reassessment

Earlier stages of the project were heavily focused on the question:

Do stored encoder hidden states preserve useful information?

Recent experiments significantly weaken this as the primary concern.

The evidence now strongly suggests:

stored hidden states
→ retain substantial semantic information

retrieved hidden states
→ remain decoder-readable

latent memories
→ survive storage and retrieval with high fidelity

The research question has therefore shifted toward:

Can the decoder reliably recover query-conditioned propositions
from externally persisted latent memories?

rather than:

Do latent memories contain information at all?
Statistical Validation Of Hidden-State Persistence

Direct comparisons were performed between:

fresh encoder outputs
cached hidden states
superposed hidden states
final latent memory presented to decoder

Observed statistics:

Fresh encoder output:

std ≈ 0.143
avg token norm ≈ 3.91

Cached memories:

std ≈ 0.148–0.151
avg token norm ≈ 4.04–4.11

Memory superposition:

std ≈ 0.166
avg token norm ≈ 4.51

Final latent passed to decoder:

std ≈ 0.160
avg token norm ≈ 4.31

Interpretation:

encoder output
≈
cached memory
≈
superposed memory
≈
decoder input

No evidence was observed for:

latent collapse
activation explosion
storage corruption
catastrophic manifold divergence

This is one of the strongest pieces of evidence obtained so far.

The latent memories remain statistically close to naturally generated encoder representations even after retrieval and composition.

Hidden-State Information Preservation Appears Strong

A critical realization emerged from later experiments.

Transformers fundamentally rely on hidden states as the communication channel between encoder and decoder.

If final encoder hidden states did not preserve rich information:

encoder-decoder transformers
would not function

Furthermore:

latent retrieval repeatedly produces document-specific concepts
proposition-level information is sometimes recovered
semantic concepts repeatedly match stored document content

Examples include generations such as:

semantic computation engine
latent semantic memory
semantic representation

which are directly related to document content.

This increasingly suggests:

information preservation
is not the dominant bottleneck
Proposition Retrieval Is Possible

An important hidden-state evaluation compared traditional retrieval against latent retrieval.

Example:

Question:

What are the advantages?

Retrieved latent memories produced:

The encoder runs once during ingestion instead of every query.

This corresponds closely to an explicit proposition present in the paper.

This result is important because it demonstrates:

document-specific proposition retrieval

rather than merely:

domain-level semantic steering

Therefore the current evidence no longer supports the strong claim that latent memories preserve only vague semantic fields.

Proposition-level information clearly survives at least in some cases.

New Interpretation Of Semantic Drift

Earlier observations were interpreted as:

hidden states preserve semantic fields
but lose exact concepts

Recent evidence suggests a more nuanced explanation.

Observed behavior:

correct document
correct domain
correct conceptual neighborhood
incorrect proposition selection

Examples:

Question:

What are encoder hidden states hypothesized to be?

Target:

reusable semantic computation artifacts

Generated answers included:

semantic computation engine
semantic understanding
encoders
a method of retrieving text

The outputs repeatedly remain inside the correct semantic region while failing to consistently recover the exact proposition.

This pattern now appears more consistent with:

query-conditioned proposition selection failure

than with:

information absence
Emerging Decoder Bottleneck Hypothesis

Current evidence increasingly supports:

memory storage
✓

memory retrieval
✓

memory persistence
✓

decoder access to memory
✓

while leaving open:

query-conditioned memory interpretation
?

The architecture stores memories that were encoded independently of future queries.

Later, the decoder receives:

query latent
+
memory latent

and must determine:

which memory regions answer the question

through cross-attention alone.

This differs fundamentally from standard encoder-decoder training where:

query
+
context

are jointly encoded before decoding begins.

The current working hypothesis is therefore:

the dominant bottleneck may be
query-conditioned latent interpretation
rather than hidden-state storage itself
Revised Bottleneck Ranking

Current ranking of likely limitations:

1. Query-conditioned latent interpretation

2. Decoder training mismatch
   (decoder never trained to read externally persisted memories)

3. Attention competition across retrieved memories

4. Latent superposition effects

5. Storage/retrieval fidelity

Storage fidelity currently has the weakest evidence as a bottleneck.

Current Working Conclusion

The project has progressed beyond demonstrating hidden-state reuse.

Evidence now strongly supports:

persistent hidden-state storage
latent retrieval
decoder consumption of latent memories
recovery of document-specific concepts
recovery of some document-specific propositions

The dominant unanswered question is no longer:

Can hidden states function as memory?

but instead:

Can a decoder trained on query-conditioned encoder outputs
reliably extract query-specific propositions
from externally persisted latent memories?

This reframes the project from a study of memory persistence into a study of:

latent memory interpretation
query-conditioned reasoning
cross-attention memory reading
and decoder-memory compatibility

which currently appears to be the most important frontier for the architecture.

Truncation Superposition Re-evaluation

Earlier truncation-based fusion (query + memory + answer) produced astronomy-related outputs ("planetary", "astronomes"), suggesting that semantic signal might survive aggressive latent fusion. However, replication on a README-based memory failed to produce consistent retrieval. Outputs were largely degenerate ("The concept of the concept...", "A re-design", etc.) despite activation statistics remaining within reasonable ranges (avg token norm ≈ 2.3, std ≈ 0.084), indicating that failure was not caused by norm explosion.

This weakens the hypothesis that truncation superposition is a reliable latent composition mechanism. The astronomy result may have been an isolated manifold coincidence or domain-level semantic amplification rather than evidence of robust memory composition.

Current evidence continues to favor sequence-wise hidden-state concatenation as the most reliable retrieval mechanism. Weighted superposition preserves some semantic information but progressively loses precision as additional memories are added. Pure superposition remains ineffective for factual retrieval.

Current conclusion: The primary unsolved problem is no longer whether hidden states can function as memory representations (concatenation demonstrates they can), but how to reduce decoder cross-attention token cost while preserving retrieval quality. Future experiments should focus on latent token compression and decoder alignment rather than arithmetic superposition.