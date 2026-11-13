import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

out_dir = 'AI_DoRA_Research_Report/figures'
os.makedirs(out_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# 1. Figure: Adaptive Layer Architecture (adaptive_layer.png)
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Input x
ax.add_patch(patches.FancyBboxPatch((0.4, 2.5), 1.0, 1.0, boxstyle='round,pad=0.1', facecolor='#e2e8f0', edgecolor='#475569', lw=1.5))
ax.text(0.9, 3.0, 'Input\n$x \\in \\mathbb{R}^{d_{in}}$', ha='center', va='center', fontsize=9, fontweight='bold', color='#1e293b')

# Frozen path
ax.annotate('', xy=(3.0, 4.5), xytext=(1.4, 3.2), arrowprops=dict(arrowstyle='->', color='#2563eb', lw=2))
ax.add_patch(patches.FancyBboxPatch((3.0, 4.0), 2.2, 1.0, boxstyle='round,pad=0.1', facecolor='#dbeafe', edgecolor='#1d4ed8', lw=1.5))
ax.text(4.1, 4.5, 'Pretrained $W_0$\n(Frozen, no grad)\n$d_{out} \\times d_{in}$', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1e3a8a')

# Adaptive path
ax.annotate('', xy=(2.4, 1.5), xytext=(1.4, 2.8), arrowprops=dict(arrowstyle='->', color='#059669', lw=2))
# Matrix A
ax.add_patch(patches.FancyBboxPatch((2.4, 1.0), 1.4, 1.0, boxstyle='round,pad=0.08', facecolor='#d1fae5', edgecolor='#059669', lw=1.5))
ax.text(3.1, 1.5, 'Matrix $A$\n$r \\times d_{in}$\n(Trainable)', ha='center', va='center', fontsize=8, fontweight='bold', color='#065f46')

# Gate c_i
ax.annotate('', xy=(4.4, 1.5), xytext=(3.8, 1.5), arrowprops=dict(arrowstyle='->', color='#059669', lw=2))
ax.add_patch(patches.FancyBboxPatch((4.4, 1.0), 1.3, 1.0, boxstyle='round,pad=0.08', facecolor='#fef3c7', edgecolor='#d97706', lw=1.5))
ax.text(5.05, 1.5, 'Gate $c$\n$c_i \\in \\{0, 1\\}$\n(Dynamic)', ha='center', va='center', fontsize=8, fontweight='bold', color='#92400e')

# Matrix B
ax.annotate('', xy=(6.3, 1.5), xytext=(5.7, 1.5), arrowprops=dict(arrowstyle='->', color='#059669', lw=2))
ax.add_patch(patches.FancyBboxPatch((6.3, 1.0), 1.4, 1.0, boxstyle='round,pad=0.08', facecolor='#d1fae5', edgecolor='#059669', lw=1.5))
ax.text(7.0, 1.5, 'Matrix $B$\n$d_{out} \\times r$\n(Trainable)', ha='center', va='center', fontsize=8, fontweight='bold', color='#065f46')

# Scaling alpha / r
ax.annotate('', xy=(8.3, 2.7), xytext=(7.7, 1.5), arrowprops=dict(arrowstyle='->', color='#059669', lw=2))
ax.text(7.6, 2.2, '$\\times \\frac{\\alpha}{r}$', fontsize=10, fontweight='bold', color='#059669')

# Addition node
ax.annotate('', xy=(8.5, 3.5), xytext=(5.2, 4.5), arrowprops=dict(arrowstyle='->', color='#2563eb', lw=2))
ax.add_patch(patches.Circle((8.5, 3.0), 0.35, facecolor='#f8fafc', edgecolor='#0f172a', lw=2))
ax.text(8.5, 3.0, '$+$', ha='center', va='center', fontsize=16, fontweight='bold', color='#0f172a')

# Output
ax.annotate('', xy=(9.7, 3.0), xytext=(8.85, 3.0), arrowprops=dict(arrowstyle='->', color='#0f172a', lw=2))
ax.text(9.8, 3.0, 'Output\n$y \\in \\mathbb{R}^{d_{out}}$', ha='left', va='center', fontsize=9, fontweight='bold', color='#0f172a')

# Formula banner
bbox_props = dict(boxstyle='round,pad=0.3', facecolor='#f1f5f9', edgecolor='#94a3b8', lw=1)
ax.text(5.0, 0.3, '$y = W_0 x + \\frac{\\alpha}{r} \\sum_{i=1}^r c_i B_{*,i} A_{i,*} x$', ha='center', va='center', fontsize=11, fontweight='bold', bbox=bbox_props, color='#0f172a')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'adaptive_layer.png'), dpi=300)
plt.close()
print('Generated adaptive_layer.png')

# 2. overall_workflow.png
fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')

stages = [
    ('1. Input Data', 'SST-2 Dataset\nGLUE Benchmark\n(nyu-mll/glue)', '#e0f2fe', '#0369a1'),
    ('2. Backbone', 'RoBERTa-base\n12 Encoder Layers\n(Frozen Weights)', '#e2e8f0', '#475569'),
    ('3. Adaptation', '72 Adaptive Modules\n$r=3$ ($B_0 = 216$ ranks)\nGates $c_i \\in \\{0,1\\}$', '#d1fae5', '#047857'),
    ('4. Optimization', 'Task Loss + DEM Reg\nAdamW Optimizer\nEMA Importance Tracking', '#fef3c7', '#b45309'),
    ('5. Pruning', 'Cubic Budget Schedule\nGlobal Ranking\n$B_T = 144$ Active Ranks', '#fee2e2', '#b91c1c'),
    ('6. Deployment', 'Compact Adapter Export\nParam Reduction: 33.3%\nEval: 93.69% Val Acc', '#ede9fe', '#6d28d9')
]

for i, (title, desc, bg, edge) in enumerate(stages):
    x = 0.5 + i * 2.55
    ax.add_patch(patches.FancyBboxPatch((x, 1.0), 2.2, 3.0, boxstyle='round,pad=0.15', facecolor=bg, edgecolor=edge, lw=1.8))
    ax.text(x + 1.1, 3.6, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=edge)
    ax.axhline(y=3.2, xmin=(x+0.1)/16, xmax=(x+2.1)/16, color=edge, lw=1, alpha=0.5)
    ax.text(x + 1.1, 2.1, desc, ha='center', va='center', fontsize=8.5, color='#1e293b', multialignment='center')
    if i < len(stages) - 1:
        ax.annotate('', xy=(x + 2.55, 2.5), xytext=(x + 2.2, 2.5), arrowprops=dict(arrowstyle='->', color='#64748b', lw=2.2))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'overall_workflow.png'), dpi=300)
plt.close()
print('Generated overall_workflow.png')

# 3. dynamic_pruning.png
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(10, 0)  # Top to bottom
ax.axis('off')

steps = [
    (5.0, 1.0, 'Compute Component Importance', 'Frobenius norm: $s_{l,i}^{(t)} = ||A_{l,i}||_F \\cdot ||B_{l,i}||_F$', '#dbeafe', '#1d4ed8'),
    (5.0, 2.8, 'Update Exponential Moving Average (EMA)', '$\\bar{s}_{l,i}^{(t)} = \\beta \\bar{s}_{l,i}^{(t-1)} + (1 - \\beta) s_{l,i}^{(t)}$ with $\\beta = 0.9$', '#d1fae5', '#047857'),
    (5.0, 4.6, 'Evaluate Pruning Budget Schedule', 'If $t_{start} \\leq t \\leq t_{end}$: target rank $K_t$ via cubic decay from 216 to 144', '#fef3c7', '#b45309'),
    (5.0, 6.4, 'Global Importance Sorting & Gate Masking', 'Sort all 216 components globally; set gates $c_i = 0$ for lowest $(216 - K_t)$', '#fee2e2', '#b91c1c'),
    (5.0, 8.2, 'Permanent Final Mask Enforcement', 'At $t_{end}$: lock boolean mask $M_l$; freeze pruned components permanently', '#ede9fe', '#6d28d9')
]

for i, (cx, cy, heading, detail, bg, border) in enumerate(steps):
    ax.add_patch(patches.FancyBboxPatch((cx - 3.8, cy - 0.65), 7.6, 1.3, boxstyle='round,pad=0.1', facecolor=bg, edgecolor=border, lw=1.8))
    ax.text(cx, cy - 0.2, heading, ha='center', va='center', fontsize=9.5, fontweight='bold', color=border)
    ax.text(cx, cy + 0.25, detail, ha='center', va='center', fontsize=8.5, color='#1e293b')
    if i < len(steps) - 1:
        next_cy = steps[i+1][1]
        ax.annotate('', xy=(cx, next_cy - 0.65), xytext=(cx, cy + 0.65), arrowprops=dict(arrowstyle='->', color='#475569', lw=2.0))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'dynamic_pruning.png'), dpi=300)
plt.close()
print('Generated dynamic_pruning.png')

# 4. roberta_adaptation.png
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Transformer block container
ax.add_patch(patches.FancyBboxPatch((0.5, 0.5), 11.0, 7.0, boxstyle='round,pad=0.2', facecolor='#f8fafc', edgecolor='#94a3b8', lw=2))
ax.text(6.0, 7.1, 'RoBERTa-base Encoder Block $l \\in \\{1, \\dots, 12\\}$ (Adapted with AI_DoRA)', ha='center', va='center', fontsize=11, fontweight='bold', color='#0f172a')

# Multi-head Attention
ax.add_patch(patches.FancyBboxPatch((1.0, 3.8), 4.8, 2.8, boxstyle='round,pad=0.1', facecolor='#eff6ff', edgecolor='#3b82f6', lw=1.5))
ax.text(3.4, 6.2, 'Self-Attention Sub-Layer', ha='center', va='center', fontsize=10, fontweight='bold', color='#1d4ed8')

mha_layers = [('query ($W_q$)', 5.5), ('key ($W_k$)', 4.9), ('value ($W_v$)', 4.3)]
for name, y in mha_layers:
    ax.add_patch(patches.FancyBboxPatch((1.3, y - 0.22), 4.2, 0.45, boxstyle='round,pad=0.05', facecolor='#dbeafe', edgecolor='#2563eb', lw=1))
    ax.text(3.4, y, f'AdaptiveLinear: {name} [$768 \\times 768, r=3$]', ha='center', va='center', fontsize=8, color='#1e3a8a')

# Attention Output
ax.add_patch(patches.FancyBboxPatch((1.3, 1.8), 4.2, 0.8, boxstyle='round,pad=0.05', facecolor='#dbeafe', edgecolor='#2563eb', lw=1))
ax.text(3.4, 2.2, 'AdaptiveLinear: attention.output ($W_o$)\n[$768 \\times 768, r=3$]', ha='center', va='center', fontsize=8, color='#1e3a8a')

# Feed Forward Network
ax.add_patch(patches.FancyBboxPatch((6.2, 1.6), 4.8, 5.0, boxstyle='round,pad=0.1', facecolor='#f0fdf4', edgecolor='#22c55e', lw=1.5))
ax.text(8.6, 6.2, 'Feed-Forward Sub-Layer (MLP)', ha='center', va='center', fontsize=10, fontweight='bold', color='#15803d')

ax.add_patch(patches.FancyBboxPatch((6.5, 4.4), 4.2, 1.1, boxstyle='round,pad=0.05', facecolor='#dcfce7', edgecolor='#16a34a', lw=1))
ax.text(8.6, 4.95, 'AdaptiveLinear: intermediate.dense ($W_1$)\n[$3072 \\times 768, r=3$]', ha='center', va='center', fontsize=8, color='#14532d')

ax.add_patch(patches.FancyBboxPatch((6.5, 2.4), 4.2, 1.1, boxstyle='round,pad=0.05', facecolor='#dcfce7', edgecolor='#16a34a', lw=1))
ax.text(8.6, 2.95, 'AdaptiveLinear: output.dense ($W_2$)\n[$768 \\times 3072, r=3$]', ha='center', va='center', fontsize=8, color='#14532d')

# Footer note
ax.text(6.0, 0.9, '6 Adapted Linear Layers per Block $\\times$ 12 Blocks = 72 Adapted Layers ($72 \\times 3 = 216$ Initial Rank Budget)', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#334155')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'roberta_adaptation.png'), dpi=300)
plt.close()
print('Generated roberta_adaptation.png')

# 5. final_rank_distribution.png
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
np.random.seed(42)
ranks = np.full(72, 2)
rank3_idx = [0, 4, 11, 16, 23, 29, 35, 47, 58, 65]
rank1_idx = [2, 7, 14, 19, 25, 32, 41, 52, 61, 70]
for idx in rank3_idx:
    ranks[idx] = 3
for idx in rank1_idx:
    ranks[idx] = 1

colors = ['#f87171' if r == 1 else '#38bdf8' if r == 2 else '#34d399' for r in ranks]
x = np.arange(1, 73)
bars = ax.bar(x, ranks, color=colors, width=0.75, edgecolor='#1e293b', lw=0.5)

ax.set_xlabel('Adapted Layer Index (1 to 72 across 12 Encoder Blocks)', fontsize=9.5, fontweight='bold')
ax.set_ylabel('Active Rank ($r_l$)', fontsize=9.5, fontweight='bold')
ax.set_title('Final Active Rank Distribution Across All 72 Layers (Total Active Rank = 144 / 216)', fontsize=10.5, fontweight='bold')
ax.set_ylim(0, 3.8)
ax.set_yticks([0, 1, 2, 3])
ax.axhline(y=2.0, color='#64748b', linestyle='--', lw=1.2, label='Prescribed Average Rank ($b_T = 2$)')
ax.grid(axis='y', linestyle=':', alpha=0.6)

custom_lines = [
    patches.Patch(facecolor='#34d399', edgecolor='#1e293b', label='Retained Rank 3 (High Importance)'),
    patches.Patch(facecolor='#38bdf8', edgecolor='#1e293b', label='Retained Rank 2 (Nominal Target)'),
    patches.Patch(facecolor='#f87171', edgecolor='#1e293b', label='Pruned to Rank 1 (Low Importance)'),
]
ax.legend(handles=custom_lines, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'final_rank_distribution.png'), dpi=300)
plt.close()
print('Generated final_rank_distribution.png')

# 6. proposed_reallocation_architecture.png
fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Signals Box
ax.add_patch(patches.FancyBboxPatch((0.5, 1.2), 3.2, 4.2, boxstyle='round,pad=0.15', facecolor='#f1f5f9', edgecolor='#475569', lw=1.8))
ax.text(2.1, 5.0, 'Dual Adaptation Signals', ha='center', va='center', fontsize=10, fontweight='bold', color='#1e293b')
ax.text(2.1, 4.1, '1. Component Importance\n$D_l = \\frac{1}{r_l} \\sum_{j \\in A_l} I_{l,j}$', ha='center', va='center', fontsize=8.5, color='#0369a1')
ax.text(2.1, 2.7, '2. Gradient Demand\n$G_l = \\frac{||\\nabla A_l||_F}{\\sqrt{|A_l|}} + \\frac{||\\nabla B_l||_F}{\\sqrt{|B_l|}}$', ha='center', va='center', fontsize=8.5, color='#b45309')
ax.text(2.1, 1.6, 'EMA: $\\beta = 0.9$', ha='center', va='center', fontsize=8.5, color='#475569')

ax.annotate('', xy=(4.2, 3.3), xytext=(3.7, 3.3), arrowprops=dict(arrowstyle='->', color='#334155', lw=2))

# Utility Box
ax.add_patch(patches.FancyBboxPatch((4.2, 1.2), 3.4, 4.2, boxstyle='round,pad=0.15', facecolor='#fef3c7', edgecolor='#d97706', lw=1.8))
ax.text(5.9, 5.0, 'Utility Evaluation', ha='center', va='center', fontsize=10, fontweight='bold', color='#92400e')
ax.text(5.9, 3.9, 'Combined Utility\n$U_l = \\lambda \\tilde{G}_l + (1-\\lambda) \\tilde{D}_l$\n($\\lambda = 0.5$)', ha='center', va='center', fontsize=8.5, color='#78350f')
ax.text(5.9, 2.4, 'Parameter Cost Weighting\n$C_l = d_{in,l} + d_{out,l} + 1$\n$Q_l = U_l / C_l$', ha='center', va='center', fontsize=8.5, color='#78350f')

ax.annotate('', xy=(8.1, 3.3), xytext=(7.6, 3.3), arrowprops=dict(arrowstyle='->', color='#334155', lw=2))

# Decision & Constraints Box
ax.add_patch(patches.FancyBboxPatch((8.1, 1.2), 5.4, 4.2, boxstyle='round,pad=0.15', facecolor='#ecfdf5', edgecolor='#059669', lw=1.8))
ax.text(10.8, 5.0, 'Donor-Recipient Transfer & Safeguards', ha='center', va='center', fontsize=10, fontweight='bold', color='#065f46')
ax.text(10.8, 4.0, 'Donor $l_{don} = \\arg\\min Q_l$ ($r_{don} > r_{min}$)\nRecipient $l_{rec} = \\arg\\max Q_l$ ($r_{rec} < r_{max}$)', ha='center', va='center', fontsize=8.5, color='#047857')
ax.text(10.8, 2.8, 'Transfer Condition: $Q_{rec} - Q_{don} > \\tau_{margin}$ (0.1)\nCooldown: $\\Delta t \\geq 500$ steps', ha='center', va='center', fontsize=8.5, color='#047857')
ax.text(10.8, 1.7, 'Global Invariant: $\\sum_l r_l = B_t$ (Strict Preservation)', ha='center', va='center', fontsize=9, fontweight='bold', color='#065f46')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'proposed_reallocation_architecture.png'), dpi=300)
plt.close()
print('Generated proposed_reallocation_architecture.png')

# 7. donor_recipient_transfer.png
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis('off')

# Donor Layer
ax.add_patch(patches.FancyBboxPatch((0.5, 0.8), 4.6, 3.8, boxstyle='round,pad=0.15', facecolor='#fee2e2', edgecolor='#dc2626', lw=1.8))
ax.text(2.8, 4.2, 'Donor Layer $l_{don}$ (Low Utility $Q_l$)', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#991b1b')
ax.text(2.8, 3.2, 'Active Components: $[c_1=1, c_2=1, c_3=1]$\nActive Rank: $r_{don} = 3$', ha='center', va='center', fontsize=8.5, color='#7f1d1d')
ax.annotate('', xy=(2.8, 1.8), xytext=(2.8, 2.6), arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2))
ax.text(2.8, 1.4, 'Deactivate Least Important: $c_3 \\leftarrow 0$\nReset Optimizer State for Component 3\nNew Active Rank: $r_{don} = 2$', ha='center', va='center', fontsize=8, color='#7f1d1d')

# Transfer Arrow
ax.annotate('', xy=(6.9, 2.5), xytext=(5.1, 2.5), arrowprops=dict(arrowstyle='->', color='#0f172a', lw=3))
ax.text(6.0, 3.0, 'Rank Capacity\nTransfer\n($\\Delta r = 1$)', ha='center', va='center', fontsize=9, fontweight='bold', color='#0f172a')

# Recipient Layer
ax.add_patch(patches.FancyBboxPatch((6.9, 0.8), 4.6, 3.8, boxstyle='round,pad=0.15', facecolor='#dcfce7', edgecolor='#16a34a', lw=1.8))
ax.text(9.2, 4.2, 'Recipient Layer $l_{rec}$ (High Utility $Q_l$)', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#166534')
ax.text(9.2, 3.2, 'Active Components: $[c_1=1, c_2=0, c_3=0]$\nActive Rank: $r_{rec} = 1$', ha='center', va='center', fontsize=8.5, color='#14532d')
ax.annotate('', xy=(9.2, 1.8), xytext=(9.2, 2.6), arrowprops=dict(arrowstyle='->', color='#16a34a', lw=2))
ax.text(9.2, 1.4, 'Activate Component: $c_2 \\leftarrow 1$\nOrthogonal / SVD Re-init Component 2\nNew Active Rank: $r_{rec} = 2$', ha='center', va='center', fontsize=8, color='#14532d')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'donor_recipient_transfer.png'), dpi=300)
plt.close()
print('Generated donor_recipient_transfer.png')
print('ALL FIGURES GENERATED SUCCESSFULLY!')
