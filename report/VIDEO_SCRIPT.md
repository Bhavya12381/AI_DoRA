# COMPLETE VIDEO SCRIPT & SCREEN PLAN
# DoRA Independent Implementation Project
# Target length: 10–12 minutes
# All content verified against project source files.
#
# Legend:
#   [VO]  = Voiceover (what presenter says)
#   [SCR] = Screen content (what is shown)
#   [CAP] = On-screen caption / slide text
#   [COD] = Code file and specific section to show
#   [FIG] = Figure to display
#   [TRN] = Transition instruction

===================================================================
COMPLETE SCENE-BY-SCENE VIDEO PLAN
===================================================================

──────────────────────────────────────────────────────────────────
SCENE 1  │  00:00–00:35  │  TITLE & HOOK
──────────────────────────────────────────────────────────────────

[SCR]  Clean slide: project title, university logo, date.
[CAP]  "Independent Implementation and Evaluation of
        Dynamic Rank Distribution for Parameter-Efficient
        Fine-Tuning of RoBERTa"
       Sub-caption: "Based on DoRA — ACL 2024"

[FIG]  figures/overall_workflow.png
       (show briefly as background, faded)

[VO]
"This video presents a project where I independently implemented
the core ideas from a 2024 ACL paper called DoRA —
Dynamic Rank Distribution for parameter-efficient fine-tuning.
The goal wasn't to reproduce every number in the paper.
It was to understand how the algorithm works, build it from scratch,
and run a real experiment on RoBERTa and the SST-2 dataset.
Let me walk you through everything — the paper, the math,
the implementation, and the actual results."

[TRN]  Fade to Scene 2.
[PURPOSE]  Establish scope and student-project framing immediately.


──────────────────────────────────────────────────────────────────
SCENE 2  │  00:35–01:30  │  THE PROBLEM — WHY PEFT?
──────────────────────────────────────────────────────────────────

[SCR]  Simple two-column layout.
       Left: image of large model (parameter count label).
       Right: bullet list appearing word by word.

[CAP]  Bullets appearing one at a time:
       • RoBERTa-base: 125 million parameters
       • Full fine-tuning: update ALL 125M
       • One fine-tuned copy per task
       • Memory + storage cost per task

[VO]
"Modern pretrained language models like RoBERTa have around
125 million parameters. If you want to fine-tune this model
on a new task — say, sentiment classification — traditional
full fine-tuning means updating every single one of those
125 million parameters.

That's expensive. You need a GPU with enough memory to hold
the model, the optimizer states, and the gradients all at once.
And if you want to use the same model for 10 different tasks,
you need 10 separate full-sized copies on disk.

Parameter-efficient fine-tuning, or PEFT, solves this by
keeping the pretrained weights frozen and only training a
small number of additional parameters.
The pretrained knowledge stays intact.
You only learn what's new for the task."

[TRN]  Slide in new content on the right.
[PURPOSE]  Motivate the problem honestly, no exaggeration.


──────────────────────────────────────────────────────────────────
SCENE 3  │  01:30–02:15  │  LOW-RANK ADAPTATION (LoRA BACKGROUND)
──────────────────────────────────────────────────────────────────

[SCR]  Animated equation build-up on dark background.
       Show: W_eff = W_0 + ΔW
       Then: ΔW = B × A (small B and A matrices illustrated)

[CAP]  "Low-Rank Adaptation (LoRA)"
       "W₀: Frozen  |  ΔW: Trainable low-rank update"
       "Parameters: rank × (d_in + d_out) instead of d_in × d_out"

[VO]
"The most popular PEFT approach is LoRA — Low-Rank Adaptation.
The idea is simple. Instead of updating the full weight matrix W,
you add a small correction ΔW that is constrained to be low-rank.
A rank-r matrix can be written as the product of two smaller matrices:
B of shape d-out by r, and A of shape r by d-in.

For RoBERTa-base, if a weight matrix is 768 by 768,
full fine-tuning has roughly 589,000 parameters.
With rank 3, B times A has only about 4,600 parameters.

LoRA works well, but it has one limitation:
it assigns the same rank to every layer, and fixes it for the
entire training run. Some layers might need more capacity.
Some might need almost none. LoRA can't adapt to that."

[TRN]  Wipe transition.
[PURPOSE]  Establish LoRA as the baseline before introducing DoRA's improvement.


──────────────────────────────────────────────────────────────────
SCENE 4  │  02:15–03:00  │  WHAT DOES DoRA ADD?
──────────────────────────────────────────────────────────────────

[SCR]  Split screen:
       Left: LoRA diagram (one fixed rank, all layers same)
       Right: DoRA diagram (components per layer, different amounts kept)

[CAP]  "DoRA — Dynamic Rank Distribution"
       "Key idea: Not all layers need the same rank"
       "Let importance decide which components to keep"

[VO]
"This is the core idea in the DoRA paper.
Instead of a single fixed low-rank matrix per layer,
decompose the update into individual rank-1 components,
each with its own trainable scalar gate.

During training, you measure how much each component
actually contributes to the learned transformation.
Components that matter — keep them.
Components that barely contribute — suppress them.

Do this globally across all layers, so the total rank budget
is shared across the whole network. One layer might end up
keeping all three components. Another might keep just one.
The model effectively self-organizes its parameter budget
according to actual importance."

[TRN]  Cross-fade to math.
[PURPOSE]  Connect paper motivation to key algorithmic innovation.


──────────────────────────────────────────────────────────────────
SCENE 5  │   03:00–04:00  │  MATHEMATICS: DECOMPOSITION
──────────────────────────────────────────────────────────────────

[SCR]  Equation on clean background. Build up piece by piece.

Equation 1 — shown step by step:
  Step 1: W_eff = W_0 + ΔW
  Step 2: ΔW = (α/r) Σᵢ cᵢ · bᵢ aᵢᵀ
  Step 3: In matrix form: ΔW = (α/r) · B · diag(c) · A

Then show the dimensions:
  A: [rank × d_in]
  B: [d_out × rank]
  c: [rank]    ← scalar gate per component

[COD]  src/dora/layer.py
       Zoom to AdaptiveRankLinear.__init__()
       Highlight lines defining self.A, self.B, self.c
       (lines 98–133)
       Then zoom to forward() method
       Highlight lines 184–195:
         low_rank = adapter_input @ self.A.T
         low_rank = low_rank * self.c
         low_rank = low_rank @ self.B.T
         low_rank = low_rank * self.scale

[CAP]  Show each symbol as it's explained:
       "W₀ — frozen pretrained weight (no gradient)"
       "A, B — trainable low-rank factors"
       "cᵢ — trainable scalar gate per component"
       "cᵢ = 0 → component i contributes nothing"
       "α/r — scaling factor (α=2.0, r=3)"

[VO]
"Let's look at the math. Each linear layer gets a trainable
weight update Delta-W that looks like this.

We have matrix A of shape rank by d-in,
matrix B of shape d-out by rank,
and a vector c containing one scalar gate per rank-1 component.

The full update is B times diagonal-of-c times A,
scaled by alpha divided by rank.

The key is that c_i gate. If we set it to zero,
that entire rank-1 component contributes nothing to the output.
The A and B tensors for that component still exist in memory —
they're still updated by the optimizer during the pruning phase —
but their effect on the output is completely suppressed.

Now let me show you exactly how this is implemented.

[Show layer.py]

Here in layer.py, you can see the three parameter groups:
self.A, self.B, and self.c.
A and B get Kaiming-uniform initialization.
But notice that c is initialized to all zeros —
so at the very start of training, the adapter contributes
absolutely nothing. The model starts as pure RoBERTa.

And here's the forward method. The computation goes:
input through A-transpose, element-wise multiply by c,
through B-transpose, then scale by alpha over rank.
This is exactly the equation, implemented directly."

[TRN]  Smooth fade, keep code visible on side.
[PURPOSE]  Paper equation mapped directly to implementation lines.


──────────────────────────────────────────────────────────────────
SCENE 6  │   04:00–04:40  │  MATHEMATICS: IMPORTANCE + EMA
──────────────────────────────────────────────────────────────────

[SCR]  Two equations side by side.

Equation 2:
  sᵢ = ‖ΔWᵢ‖_F / (‖Σⱼ ΔWⱼ‖_F + ε)

Equation 3:
  ŝᵢ(t) = β·ŝᵢ(t-1) + (1-β)·sᵢ(t)

[COD]  src/dora/importance.py
       Show entire function component_importance()
       Highlight:
         component_norms (lines 29–33)
         total_norm (lines 35–40)
         scores = component_norms / (total_norm + eps)  (lines 42–44)

[COD]  src/dora/pruner.py
       Show update_importance() method
       Highlight:
         updated = ema_decay * previous + (1.0 - ema_decay) * current
         (lines 110–116)

[CAP]  "Importance: how much does component i contribute
        to the total weight update?"
       "EMA smoothing prevents noisy pruning decisions"
       "β = 0.9 (smoothing factor)"

[VO]
"How do we decide which components to prune?
We measure each component's relative contribution to the
total weight update using Frobenius norms.

The importance score s_i is the Frobenius norm of component i's
rank-1 weight matrix, divided by the Frobenius norm of the
full combined update. So it's a relative measure —
how much does this component contribute compared to the whole?

There's a small epsilon in the denominator for numerical stability.
At initialization, all gates are zero, so all importance scores
would be zero divided by zero. The epsilon prevents that.

But raw importance scores computed at a single training step
are noisy. So we smooth them with an exponential moving average.
Beta of 0.9 means the new score contributes only 10% of the update,
so the estimate responds gradually rather than jumping around.

[Show importance.py]

Here in importance.py, you can see this directly.
Component matrices are computed, their Frobenius norms taken,
the total update norm computed, and scores returned.

[Show pruner.py update_importance]

And here in the pruner, the EMA update:
new score equals 0.9 times previous, plus 0.1 times current."

[TRN]  Slide to cubic schedule scene.
[PURPOSE]  Two equations shown, both connected to exact implementation lines.


──────────────────────────────────────────────────────────────────
SCENE 7  │   04:40–05:20  │  CUBIC PRUNING SCHEDULE + GLOBAL PRUNING
──────────────────────────────────────────────────────────────────

[SCR]  Left: show rank_schedule.png figure.
       Right: equation overlaid.

Equation 4 (shown alongside figure):
  b(t) = b₀ − (b₀ − bT) × ξ³
  where ξ = (t − t_start) / (t_end − t_start)

Label the three zones on the figure:
  [Warmup: flat at 216]  [Cubic decay: 216→144]  [Fixed: flat at 144]

Key values shown on screen:
  t_start = 9,477 steps (15% of 63,180)
  t_end   = 31,590 steps (50% of 63,180)
  b₀      = 3 (initial average rank per layer)
  bT      = 2 (target final average rank)

[COD]  src/dora/scheduler.py
       Show budget() method (lines 51–69)
       Highlight the three if/elif blocks
       Highlight: cubic_progress = progress ** 3

[COD]  src/dora/pruner.py
       Show prune() method — the argsort section
       Highlight lines 412–418:
         sorted_positions = torch.argsort(all_scores, stable=True)
         positions_to_prune = set(sorted_positions[:prune_rank_num].tolist())

[CAP]  "Three phases of training:"
       "1. Warmup (t < 9,477): rank held at 216"
       "2. Cubic decay (9,477 → 31,590): 216 → 144"
       "3. Stable (t > 31,590): rank locked at 144"

[VO]
"The pruning doesn't happen all at once.
It follows a three-phase schedule.

In the warmup phase — the first 15% of training —
no pruning happens. The adapters are learning from scratch
and we don't want to prune before they've had a chance to develop.

Then the cubic decay phase begins. Over the next 35% of training,
the target total rank decreases from 216 down to 144.
The cubic shape means the reduction is fast at first,
then slows as we approach the target. That gives the model
time to adapt near the end rather than making a sudden jump.

After that, pruning stops. The final budget of 144 is locked in.

[Show the figure]

You can see this clearly in our rank-schedule figure, which was
generated directly from the scheduler code.

[Show scheduler.py]

Here's the budget function. Three if branches.
Before start: return initial rank.
After end: return final rank.
In between: compute cubic interpolation with progress cubed.

Now the important question: which components get pruned?
The answer is: the globally lowest-scoring ones.

[Show pruner.py argsort section]

All 216 EMA scores — one per component across all 72 layers —
are concatenated into a single vector.
We sort them with argsort using stable=True for deterministic
tie-breaking, and take the bottom however-many-we-need-to-remove.
Those are the ones whose gates get set to zero."

[FIG]  figures/rank_schedule.png
       Show for approximately 25 seconds while explaining.
[TRN]  Cross-fade to DEM scene.
[PURPOSE]  Schedule figure + code side by side, cubic formula explained.


──────────────────────────────────────────────────────────────────
SCENE 8  │   05:20–05:55  │  DEM REGULARIZATION
──────────────────────────────────────────────────────────────────

[SCR]  Equation:
  L_DEM = (1/Nc) × Σ_ℓ Σᵢ (Var(Aᵢ) + Var(Bᵢ))
  L_total = L_task + η × L_DEM
  where Nc = 2 × r × L = 2 × 3 × 72 = 432

[COD]  src/dora/regularization.py
       Show dem_regularization() function
       Highlight:
         a_variance = torch.var(layer.A, dim=1, unbiased=True)
         b_variance = torch.var(layer.B, dim=0, unbiased=True)
         total_variance / float(component_count)
       Show dem_loss() showing:
         total_loss = task_loss + coefficient * regularization

[CAP]  "DEM = Dimensional Equilibrium Modulator"
       "Penalizes low-variance factor rows/columns"
       "Prevents components collapsing to same direction"
       "η = 0.5 in our experiment"
       "Nc = 432 (normalization factor)"

[VO]
"There's one more piece: DEM regularization.
The paper calls it the Dimensional Equilibrium Modulator.
The problem it solves is this: if multiple rank-1 components
learn to point in the same direction, some of them become
redundant. They might all get high importance scores
but they're all doing the same thing.

DEM fixes this by penalizing low-variance factor matrices.
For each row of A and each column of B, it computes the
unbiased variance across the elements in that row or column.
Low variance means the values are all similar —
the component has collapsed. A high penalty pushes them apart.

[Show regularization.py]

Here's the implementation. Notice it computes variance
along dimension 1 for A — that's across each row —
and dimension 0 for B — that's across each column.
The total is normalized by the count of all components
across all layers, which in our case is 432.

The combined loss is simple: task loss plus 0.5 times DEM.
Both terms flow gradients back to A and B."

[TRN]  Wipe to architecture section.
[PURPOSE]  DEM equation connected directly to regularization.py lines.


──────────────────────────────────────────────────────────────────
SCENE 9  │   05:55–07:00  │  OUR ROBERTA IMPLEMENTATION
──────────────────────────────────────────────────────────────────

[SCR]  Show roberta_adaptation.png figure on left half.
       On right: text list builds up line by line.

[CAP]  "RoBERTa-base: 12 encoder layers"
       "Per layer — 6 adapted projections:"
       "  1. Query"
       "  2. Key"
       "  3. Value"
       "  4. Attention output"
       "  5. Intermediate dense"
       "  6. Output dense"
       "→ 12 × 6 = 72 adaptive layers"
       "→ 72 × 3 = 216 initial rank components"

[FIG]  figures/roberta_adaptation.png
       Show for ~30 seconds while listing the six targets.

[COD]  examples/hf_roberta_sst2.py
       Show the target_names construction loop (lines 427–442):
         for layer_idx in range(12):
             prefix = f"roberta.encoder.layer.{layer_idx}"
             target_names.extend([
                 f"{prefix}.attention.self.query",
                 ...
             ])

[COD]  src/dora/transformer.py
       Show freeze_model() (lines 6–15):
         for parameter in model.parameters():
             parameter.requires_grad = False

       Then show replace_linear_layers() (lines 18–108)
       Highlight lines 77–92:
         adapter = AdaptiveRankLinear(base_layer=module, ...)
         adapter.A.requires_grad = True
         adapter.B.requires_grad = True
         adapter.c.requires_grad = True
         adapter.base.weight.requires_grad = False

[VO]
"Now let's look at how we set this up with RoBERTa.

RoBERTa-base has 12 Transformer encoder layers.
Each layer contains six linear transformations:
query, key, and value for the self-attention,
the output projection after attention,
and two projections in the feed-forward block.

So 12 layers times 6 projections equals 72 adaptive layers.
With rank 3 each, that's 216 total initial rank components.

[Show architecture figure]

[Show hf_roberta_sst2.py target names loop]

Here in the training script, you can see the exact list
of target layer names being constructed for all 12 encoder blocks.

[Show transformer.py freeze_model]

The first thing we do is freeze the entire model.
Every parameter gets requires_grad set to False.
The pretrained knowledge is locked.

[Show replace_linear_layers]

Then we go through and replace each target linear module
with an AdaptiveRankLinear. Notice that the adapter's A, B,
and c parameters are explicitly set trainable.
But the base weight stays frozen.
This is the PEFT setup — only the small adapter parameters
are updated during training."

[TRN]  Smooth slide to training loop.
[PURPOSE]  Architecture figure + two code files showing freeze + inject pattern.


──────────────────────────────────────────────────────────────────
SCENE 10  │  07:00–07:45  │  TRAINING LOOP WALKTHROUGH
──────────────────────────────────────────────────────────────────

[SCR]  Flowchart animation (can re-use overall_workflow.png).
       Arrows light up in sequence as each step is mentioned.

[CAP]  Step-by-step labels appearing:
       "SST-2 batch → tokenizer → frozen RoBERTa"
       "→ adaptive layers → task loss"
       "→ + DEM regularization → combined loss"
       "→ backprop → AdamW → optimizer.step()"
       "→ importance update → EMA → pruning check"
       "→ enforce_final_mask() → next step"

[COD]  examples/hf_roberta_sst2.py
       Show the inner training loop (lines 559–636):
         outputs = model(**batch)
         task_loss = outputs.loss
         loss, regularization = dem_loss(task_loss, model, DEM_COEFFICIENT)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         result = pruner.step(step)
         pruner.enforce_final_mask()

[VO]
"The training loop follows a straightforward sequence.

A mini-batch of SST-2 examples comes in, gets tokenized,
passes through the frozen RoBERTa backbone and the adaptive layers,
and produces logits.

The cross-entropy task loss is computed.
Then the DEM regularization term is added.
We backpropagate the combined loss and run the AdamW optimizer.

After the optimizer step, we call pruner.step().
This does two things: updates the EMA importance scores,
and — if we're in the pruning window and at a pruning checkpoint —
identifies and suppresses the lowest-scoring components.

Then we call enforce_final_mask() every single step.
During the pruning phase, this does nothing.
But once pruning finishes, it re-zeros any gate that the
optimizer might have accidentally updated away from zero.

[Show hf_roberta_sst2.py inner loop]

Here's the actual training loop. Notice the exact sequence:
dem_loss, backward, optimizer.step,
then pruner.step, then enforce_final_mask.
This order is not arbitrary — it's required for correctness."

[FIG]  figures/overall_workflow.png
       Show throughout this scene.
[TRN]  Fade to results.
[PURPOSE]  Show actual training loop, explain the call order matters.


──────────────────────────────────────────────────────────────────
SCENE 11  │   07:45–08:15  │  CHECKPOINT & EVALUATION
──────────────────────────────────────────────────────────────────

[SCR]  Terminal/notebook output showing checkpoint loading.
       Show the --evaluate flag command.

[CAP]  "Final checkpoint: epoch 60, step 63,180"
       "Evaluation-only mode (no retraining)"

[COD]  Terminal window in Colab:
       python examples/hf_roberta_sst2.py --evaluate

       Then show the printed output:
         Evaluation checkpoint: /content/drive/...
         Device: cuda
         Checkpoint epoch: 60
         Checkpoint global step: 63180
         Total active rank: 144
         Target final total rank: 144
         Training accuracy: 99.74%
         Validation accuracy: 93.69%

[VO]
"After 60 epochs, the training was complete.
The final checkpoint was saved to Google Drive automatically
after every epoch during training.
To evaluate, I didn't retrain anything.
I ran the script in evaluation mode with the --evaluate flag.

This loads the checkpoint, reconstructs the model architecture,
restores the pruner's final mask, calls enforce_final_mask()
to ensure the pruned topology is active, and then runs
evaluation on both the full training set and the validation set.

No backpropagation, no optimizer updates.
Pure inference."

[TRN]  Zoom transition to results slide.
[PURPOSE]  Show the checkpoint/resume system, --evaluate command, honest evaluation description.


──────────────────────────────────────────────────────────────────
SCENE 12  │   08:15–09:00  │  ACTUAL RESULTS
──────────────────────────────────────────────────────────────────

[SCR]  Large results display.
       Split into two panels:

       Panel 1 — Rank Results:
         Initial total rank: 216
         Final total active rank: 144
         Reduction: 72 components (33.3%)
         Non-uniform distribution across 72 layers

       Panel 2 — Accuracy Results:
         Training accuracy: 99.74%
         Validation accuracy: 93.69%
         Gap: 6.05 percentage points

[FIG]  figures/accuracy_comparison.png (large, centered)
       Duration: ~25 seconds.

[FIG]  figures/final_rank_distribution.png
       (if available from checkpoint extraction)
       Duration: ~20 seconds.

[CAP]  "Training Accuracy: 99.74%"
       "Validation Accuracy: 93.69%"
       "Final Active Rank: 144 / 216"
       "Rank reduced by exactly 33.3%"
       "Distribution is NON-UNIFORM across 72 layers"

[VO]
"Here are the actual results from our experiment.

First, the rank budget. We started with 216 total active components.
The pruning schedule targeted a final budget of 144.
And the final measurement confirmed exactly 144 active components.
The system hit the target precisely.

That's a reduction of 72 components, or one-third of the initial budget.

Importantly, the distribution is non-uniform.
The 144 retained components are not evenly split across 72 layers —
it's not simply 2 per layer. Some layers retained all 3.
Some retained just 1 or even 0.
The global ranking naturally allocated more capacity to layers
where the components had higher measured importance.

On accuracy:
Training accuracy was 99.74%.
Validation accuracy was 93.69%.

[Show accuracy figure]

That's a gap of about 6 percentage points.
The model has clearly learned the training set very well,
but there's a visible gap to unseen examples.
I'll discuss what that suggests in a moment."

[TRN]  Gentle wipe to discussion.
[PURPOSE]  Show verified numbers, rank figure, accuracy figure, explain non-uniform.


──────────────────────────────────────────────────────────────────
SCENE 13  │   09:00–09:45  │  DISCUSSION & LIMITATIONS
──────────────────────────────────────────────────────────────────

[SCR]  Two-column layout.
       Left: "What worked"   Right: "What's missing / uncertain"

[CAP]  Left column appears first:
       ✓ Dynamic rank budget hit exactly
       ✓ Global non-uniform allocation
       ✓ EMA, cubic schedule, mask enforcement
       ✓ Checkpoint + resume worked over 60 epochs
       ✓ 93.69% validation accuracy on SST-2

       Right column:
       ✗ No LoRA / full fine-tuning baseline
       ✗ No multi-seed results (single run)
       ✗ A/B tensors not physically deleted
       ✗ No inference speedup measured
       ✗ Training accuracy gap unexplained

[VO]
"Let me be honest about what this project achieved and
what it didn't.

The dynamic rank allocation worked correctly.
The system reached exactly the target budget,
the distribution is non-uniform as expected,
and all the core components — EMA, cubic schedule,
importance scoring, final masking — behave as designed.
The checkpoint system held up over 60 epochs of training.
The 93.69% validation accuracy is a reasonable result.

But there are real limitations.

First, there's no comparison baseline. I didn't run the same
setup with fixed-rank LoRA or full fine-tuning.
So I can't claim this is better or worse than alternatives.

Second, this is a single training run. One seed.
For any serious conclusion about accuracy, you'd need
multiple runs to characterize the variance.

Third, and this is important — the rank pruning in this
implementation is gate-based, not tensor-based.
The A and B matrices for pruned components still exist in GPU memory.
Setting the gate to zero suppresses their effect functionally,
but doesn't save actual memory or computation yet.
Physical compression would need a separate weight restructuring step.

Finally, the 6-point training-validation gap suggests overfitting.
The learning rate may be slightly high, and there's no
learning rate schedule. These are things to tune in future work."

[TRN]  Fade to future work.
[PURPOSE]  Honest, balanced discussion. Explicitly states what wasn't done.


──────────────────────────────────────────────────────────────────
SCENE 14  │  09:45–10:30  │  FUTURE WORK & CONCLUSION
──────────────────────────────────────────────────────────────────

[SCR]  Clean slide. Two sections.

Section 1 — Future Experiments:
  • Compare against fixed-rank LoRA baseline
  • Test multiple rank budgets (1, 2, 4, 8)
  • Multi-seed evaluation
  • Try more GLUE tasks
  • Add learning rate warmup schedule
  • Measure actual GPU memory + latency
  • Implement physical tensor compaction

Section 2 — Final Summary.

[CAP]  Final summary text (appears piece by piece):
       "Paper: DoRA — ACL 2024"
       "Idea: importance-based dynamic rank allocation"
       "We implemented: 8 core algorithmic components"
       "Experiment: RoBERTa-base, SST-2, 60 epochs, Tesla T4"
       "Result: 93.69% validation accuracy"
       "Rank: 216 → 144 (target met exactly)"
       "Educational implementation — not a paper reproduction"

[VO]
"There's plenty of interesting future work.

The most obvious next step would be running a direct comparison —
same model, same dataset, fixed rank with LoRA versus dynamic rank.
That would tell us whether the adaptive allocation actually helps.

Testing multiple final rank budgets would also reveal whether
2 per layer is optimal, or whether a different target
gives better accuracy for the same parameter count.

And physically compacting the weight matrices after pruning
would be needed to get actual inference speedup.

Let me wrap up with what this project actually is.

The paper proposes decomposing low-rank adapters into
individual rank-1 components, measuring their importance
during training with normalized Frobenius norms and EMA,
and using a cubic schedule to progressively reduce the
rank budget to a globally optimized allocation.

In this project, I implemented these ideas independently:
the adaptive layer, the importance calculation, the EMA tracking,
the cubic scheduler, the global pruner, the final mask,
and the DEM regularization.

I ran this on RoBERTa-base for 60 epochs on SST-2.
The final result was 93.69% validation accuracy,
with the rank budget hitting exactly 144 of 216 components.

This is an educational implementation.
I'm not claiming to reproduce the paper's benchmark numbers.
What I am claiming is that the algorithm works as described,
the implementation is correct, the results are real,
and I learned a lot about how dynamic rank allocation actually functions."

[TRN]  Fade to final title card.
[PURPOSE]  Honest conclusion, clearly distinguishes educational implementation.


──────────────────────────────────────────────────────────────────
SCENE 15  │  10:30–11:00  │  CLOSING TITLE CARD
──────────────────────────────────────────────────────────────────

[SCR]  Return to title card design from Scene 1.
       Add project stats as bullet points:
         72 adaptive layers
         216 → 144 rank components
         99.74% training / 93.69% validation
         60 epochs on Tesla T4

[CAP]  "Thank you"
       "Project code: AI_DoRA/"
       "Based on: DoRA, ACL 2024"
       (Mao et al. / ACL 2024 citation)

[VO]
"Thanks for watching. The full project code, report, and
all the figures shown here are in the AI_DoRA directory.
The reference for the original paper is in the description."

[TRN]  Fade out.
[PURPOSE]  Clean close with honest attribution.


===================================================================
SCREEN RECORDING PLAN
===================================================================

RECORDING 1: Title and project overview
────────────────────────────────────────
File:      report/main.tex (or a clean title slide)
Section:   Title, author, abstract first paragraph
Action:    Open in Overleaf or text editor. Slowly scroll
           through the title and abstract. No code yet.
Duration:  30 seconds
Notes:     Can also use a pre-made PowerPoint slide.


RECORDING 2: AdaptiveRankLinear layer definition
──────────────────────────────────────────────────
File:      src/dora/layer.py
Section:   Class AdaptiveRankLinear (lines 5–272)
Zoom:      First into __init__() — highlight self.A (line 98),
           self.B (line 111), self.c (line 127)
           Emphasize: self.c initialized to zeros (line 128)
           Then zoom to reset_parameters() (line 139)
           Then forward() — highlight lines 184–195
Highlight: The three lines:
           low_rank = adapter_input @ self.A.T
           low_rank = low_rank * self.c
           low_rank = low_rank @ self.B.T
Duration:  45 seconds
Notes:     Use VS Code or similar with syntax highlighting.
           Increase font size to minimum 16pt for readability.


RECORDING 3: Component importance calculation
──────────────────────────────────────────────
File:      src/dora/importance.py
Section:   component_importance() function (lines 5–46)
Zoom:      Highlight:
           component_norms (lines 29–33) — "Frobenius norm per component"
           total_update = components.sum(dim=0) (line 35) — "sum first"
           total_norm (lines 37–40) — "norm of the combined update"
           scores = component_norms / (total_norm + eps) (lines 42–44)
Duration:  30 seconds


RECORDING 4: EMA update in pruner
────────────────────────────────────
File:      src/dora/pruner.py
Section:   update_importance() method (lines 94–117)
Zoom:      Highlight:
           current = component_importance(layer).detach()
           updated = self.ema_decay * previous + (1.0 - self.ema_decay) * current
           self.ema_scores[name] = updated.detach().cpu()
Duration:  20 seconds
Notes:     Point out ema_decay = 0.9


RECORDING 5: Cubic budget scheduler
──────────────────────────────────────
File:      src/dora/scheduler.py
Section:   budget() method (lines 51–69)
Zoom:      All three branches:
           if step <= self.start_step: return initial
           if step >= self.end_step: return final
           The cubic interpolation block (lines 58–69)
           Highlight: cubic_progress = progress ** 3
Duration:  25 seconds


RECORDING 6: Global pruning — argsort selection
──────────────────────────────────────────────────
File:      src/dora/pruner.py
Section:   prune() method — the argsort section (lines 404–455)
Zoom:      Highlight:
           prune_rank_num = min(desired_removed, all_scores.numel())
           sorted_positions = torch.argsort(all_scores, stable=True)
           positions_to_prune = set(sorted_positions[:prune_rank_num].tolist())
           The loop: layer.prune_components([component_index])
Duration:  30 seconds


RECORDING 7: Final mask enforcement
──────────────────────────────────────
File:      src/dora/pruner.py
Section:   enforce_final_mask() (lines 288–303)
           And final mask recording in prune() (lines 461–479)
Zoom:      Show:
           if not self.pruning_finished: return
           self.apply_final_mask()
           The final_mask recording block
Duration:  20 seconds


RECORDING 8: DEM regularization
─────────────────────────────────
File:      src/dora/regularization.py
Section:   dem_regularization() (lines 6–76) and dem_loss() (lines 79–106)
Zoom:      Highlight:
           a_variance = torch.var(layer.A, dim=1, unbiased=True)
           b_variance = torch.var(layer.B, dim=0, unbiased=True)
           total_variance / float(component_count)
           Then: total_loss = task_loss + coefficient * regularization
Duration:  25 seconds


RECORDING 9: Model freeze and layer replacement
─────────────────────────────────────────────────
File:      src/dora/transformer.py
Section:   freeze_model() (lines 6–15) and replace_linear_layers() (lines 18–108)
Zoom:      freeze_model: "parameter.requires_grad = False"
           replace_linear_layers: AdaptiveRankLinear() call,
           then lines 90–98 showing A/B/c.requires_grad = True,
           base.weight.requires_grad = False
Duration:  30 seconds


RECORDING 10: Training script — configuration
───────────────────────────────────────────────
File:      examples/hf_roberta_sst2.py
Section:   Configuration block (lines 41–83)
Zoom:      Show all constant definitions:
           BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
           INITIAL_RANK, FINAL_RANK, ALPHA, DEM_COEFFICIENT,
           EMA_DECAY, START_FRACTION, END_FRACTION
Duration:  20 seconds


RECORDING 11: Training script — target layer names
─────────────────────────────────────────────────────
File:      examples/hf_roberta_sst2.py
Section:   Target layer construction loop (lines 425–450)
Zoom:      The for loop over range(12) building target_names,
           showing all 6 projection types per block
Duration:  20 seconds


RECORDING 12: Training script — inner training loop
─────────────────────────────────────────────────────
File:      examples/hf_roberta_sst2.py
Section:   Inner per-batch loop (lines 559–636)
Zoom:      Highlight in sequence:
           outputs = model(**batch)
           task_loss = outputs.loss
           loss, regularization = dem_loss(...)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           result = pruner.step(step)
           pruner.enforce_final_mask()
Duration:  35 seconds


RECORDING 13: Evaluation command + output
───────────────────────────────────────────
Location:  Colab terminal or pre-recorded terminal output
Command:   python examples/hf_roberta_sst2.py --evaluate
Show:      The printed output including:
           Checkpoint epoch: 60
           Checkpoint global step: 63180
           Total active rank: 144
           Training accuracy: 99.74%
           Validation accuracy: 93.69%
Duration:  25 seconds
Notes:     If re-running is not feasible, show a screenshot
           of the original terminal output.


RECORDING 14: Rank schedule figure
─────────────────────────────────────
File:      report/figures/rank_schedule.png
Action:    Open full-screen. Slowly pan:
           First to the flat section at 216,
           Then to the cubic decay curve,
           Then to the flat section at 144.
           Point out t_start and t_end dashed lines.
Duration:  25 seconds


RECORDING 15: Accuracy comparison figure
──────────────────────────────────────────
File:      report/figures/accuracy_comparison.png
Action:    Open full-screen. Hold still.
           Let the two bars speak clearly.
Duration:  15 seconds


RECORDING 16: Final rank distribution (if available)
──────────────────────────────────────────────────────
File:      report/figures/final_rank_distribution.png
Action:    Open full-screen. Pan slowly across all 72 bars.
           Point out bars that differ in height.
Duration:  20 seconds
Notes:     If the checkpoint has not yet been extracted,
           show the README_figures.md placeholder and explain
           how to generate it from the checkpoint.


===================================================================
SHORT REHEARSAL SCRIPT (condensed version — ~4 minutes)
===================================================================

"In this project, I implemented the core ideas from the DoRA paper
published at ACL 2024.

The problem is simple: fine-tuning large language models updates
every parameter, which is expensive. LoRA adds low-rank matrices
instead, but assigns the same rank to every layer.
DoRA goes further — it decomposes each low-rank update into
individual rank-1 components, measures their importance during
training, and progressively removes the least important ones.

The math: the weight update is a sum of rank-1 outer products,
each multiplied by a trainable scalar gate c_i.
Setting c_i to zero removes that component's effect.
Importance is the Frobenius norm of each component relative to
the total update. That's smoothed with EMA — beta 0.9 —
to prevent noisy pruning decisions.

The pruning follows a three-phase schedule. Warmup, cubic decay,
then a stable final budget. In our project the total rank
goes from 216 down to 144 over the first 50% of training.

We also use DEM regularization — variance of the A and B factors —
to prevent components from collapsing to the same direction.

For the experiment: RoBERTa-base, SST-2, 60 epochs on a Tesla T4.
72 adaptive layers, initial rank 3 per layer.

Final result: the rank budget hit exactly 144, non-uniformly
distributed across all 72 layers. Validation accuracy: 93.69%.
Training accuracy: 99.74%.

The main limitations: no comparison baseline, one training run,
and the pruning is gate-based — the A/B tensors still exist in memory.
Physical compression would need a separate step.

This is an educational implementation. The algorithm works
as the paper describes, and the experiment demonstrates
the key behavior — dynamic, globally-optimized rank allocation —
in a real RoBERTa fine-tuning setting."


===================================================================
COMPLETE ON-SCREEN TEXT / CAPTION REFERENCE
===================================================================

Scene 1:  "DoRA: Dynamic Rank Distribution"
          "Independent Implementation & Evaluation"
          "RoBERTa-base + SST-2"

Scene 2:  "RoBERTa-base: ~125M parameters"
          "Full fine-tuning: update ALL parameters"
          "PEFT: train only new adapter parameters"

Scene 3:  "LoRA: W = W₀ + B·A"
          "Rank-r update: r << d_in, d_out"
          "Problem: fixed rank, same for every layer"

Scene 4:  "DoRA: decompose ΔW into rank-1 components"
          "Gate cᵢ controls each component independently"
          "Globally prune least important components"

Scene 5:  "ΔW = (α/r) · Σᵢ cᵢ · bᵢaᵢᵀ"
          "A: [3 × d_in]  B: [d_out × 3]  c: [3]"
          "cᵢ = 0 → component i inactive"
          "Initialized: c = [0, 0, 0]"

Scene 6:  "sᵢ = ‖ΔWᵢ‖_F / (‖Σⱼ ΔWⱼ‖_F + ε)"
          "ŝᵢ(t) = 0.9·ŝᵢ(t-1) + 0.1·sᵢ(t)"
          "ε = 1e-12 (numerical stability)"

Scene 7:  "Phase 1: t < 9,477 → rank = 216"
          "Phase 2: cubic decay → 216 to 144"
          "Phase 3: t > 31,590 → rank = 144"
          "Global argsort: lowest scored → pruned"

Scene 8:  "L_DEM = (1/432) Σ (Var(A) + Var(B))"
          "L_total = L_task + 0.5 × L_DEM"

Scene 9:  "12 encoder layers × 6 projections = 72 layers"
          "freeze_model() → requires_grad = False"
          "replace_linear_layers() → AdaptiveRankLinear"
          "Only A, B, c are trainable"

Scene 10: "Training step order:"
          "forward → task loss → DEM → combined loss"
          "→ backward → optimizer.step()"
          "→ pruner.step() → enforce_final_mask()"

Scene 11: "python examples/hf_roberta_sst2.py --evaluate"
          "No retraining. Checkpoint loaded from Drive."
          "enforce_final_mask() applied before evaluation"

Scene 12: "Initial total rank: 216"
          "Final total active rank: 144"
          "Rank reduction: 33.3%"
          "Training accuracy: 99.74%"
          "Validation accuracy: 93.69%"
          "Gap: 6.05 percentage points"
          "Distribution: NON-UNIFORM across 72 layers"

Scene 13: "✓ Rank budget met exactly"
          "✓ Non-uniform allocation (global pruning works)"
          "✗ No comparison baseline"
          "✗ Single training run (no multi-seed)"
          "✗ Gate-based pruning only (not tensor-compressed)"

Scene 14: "Future: baseline comparison, multi-seed, latency measurement"
          "This is an educational implementation."
          "Algorithm: verified ✓  Experiment: completed ✓"

Scene 15: "Thank you"
          "Project: AI_DoRA/"
          "Paper: DoRA, ACL 2024 (Mao et al.)"
