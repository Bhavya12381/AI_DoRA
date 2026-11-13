import os
import sys
from pathlib import Path

out_dir = Path("presentation")
out_dir.mkdir(parents=True, exist_ok=True)

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

import docx
from docx.shared import Inches as DocxInches, Pt as DocxPt, RGBColor as DocxRGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# --- Color Constants ---
BG_COLOR = RGBColor(248, 250, 252)       # Slate 50
NAVY = RGBColor(15, 23, 42)              # Slate 900
DARK_BLUE = RGBColor(30, 41, 59)         # Slate 800
WHITE = RGBColor(255, 255, 255)
CARD_BG = RGBColor(255, 255, 255)
CARD_BORDER = RGBColor(203, 213, 225)    # Slate 300
TEXT_MAIN = RGBColor(15, 23, 42)
TEXT_MUTED = RGBColor(71, 85, 105)       # Slate 600
ACCENT_BLUE = RGBColor(2, 132, 199)      # Sky 600
ACCENT_TEAL = RGBColor(13, 148, 136)     # Teal 600
ACCENT_AMBER = RGBColor(217, 119, 6)     # Amber 600
ACCENT_ROSE = RGBColor(225, 29, 72)      # Rose 600
CALLOUT_BG = RGBColor(239, 246, 255)     # Blue 50
CALLOUT_BORDER = RGBColor(147, 197, 253) # Blue 300


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5)
        )
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, category_text, title_text, subtitle_text, slide_num):
        set_slide_background(slide)
        
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8.5), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE
        p_cat.font.name = "Arial"

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.0), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY
        p_title.font.name = "Arial"

        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.5), Inches(0.4))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle_text
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = TEXT_MUTED
        p_sub.font.name = "Calibri"

        footer_box = slide.shapes.add_textbox(Inches(10.0), Inches(7.05), Inches(2.5), Inches(0.3))
        tf_f = footer_box.text_frame
        p_f = tf_f.paragraphs[0]
        p_f.alignment = PP_ALIGN.RIGHT
        p_f.text = f"Slide {slide_num} of 10"
        p_f.font.size = Pt(10)
        p_f.font.bold = True
        p_f.font.color.rgb = TEXT_MUTED
        p_f.font.name = "Calibri"

        proj_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(7.0), Inches(0.3))
        tf_p = proj_box.text_frame
        p_p = tf_p.paragraphs[0]
        p_p.text = "AI_DoRA: Parameter-Efficient Fine-Tuning & Dynamic Rank Reallocation"
        p_p.font.size = Pt(9.5)
        p_p.font.color.rgb = TEXT_MUTED
        p_p.font.name = "Calibri"

    def add_card(slide, left, top, width, height, title="", bg_color=CARD_BG, border_color=CARD_BORDER, border_width=1):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(border_width)
        else:
            card.line.fill.background()
        
        if title:
            t_box = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.18), width - Inches(0.4), Inches(0.4))
            tf = t_box.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(13.5)
            p.font.bold = True
            p.font.color.rgb = NAVY
            p.font.name = "Arial"
        return card

    # =========================================================================
    # SLIDE 1: Background: Parameter-Efficient Fine-Tuning
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    add_header(
        s1,
        "Part I: Existing Research & Problem Statement",
        "Background: Parameter-Efficient Fine-Tuning",
        "Addressing computational and memory bottlenecks when adapting large pretrained language models",
        1
    )

    add_card(s1, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.3), "Full Fine-Tuning (Conventional)")
    tb1_l = s1.shapes.add_textbox(Inches(1.0), Inches(2.35), Inches(5.2), Inches(3.6))
    tf1_l = tb1_l.text_frame
    tf1_l.word_wrap = True
    
    bullets_s1_l = [
        ("Massive Parameter Footprint: ", "Updates all parameters W_0 in large language models (e.g. 125M in RoBERTa-base, billions in LLMs)."),
        ("Prohibitive Optimizer Memory: ", "AdamW optimizer tracks 1st and 2nd gradient moments for every parameter, multiplying GPU VRAM requirements 4-6x beyond raw weights."),
        ("Deployment Storage Explosion: ", "Serving N downstream tasks requires storing N full copies of the model (hundreds of gigabytes or terabytes)."),
        ("Optimization Instability: ", "Updating all weights risks representation collapse, overfitting on smaller domain datasets, and catastrophic forgetting.")
    ]
    for i, (bold_txt, norm_txt) in enumerate(bullets_s1_l):
        p = tf1_l.paragraphs[0] if i == 0 else tf1_l.add_paragraph()
        p.space_after = Pt(10)
        r1 = p.add_run()
        r1.text = "• " + bold_txt
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = norm_txt
        r2.font.size = Pt(11.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s1, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.3), "Parameter-Efficient Fine-Tuning (PEFT / LoRA)")
    tb1_r = s1.shapes.add_textbox(Inches(7.1), Inches(2.35), Inches(5.2), Inches(3.6))
    tf1_r = tb1_r.text_frame
    tf1_r.word_wrap = True
    
    bullets_s1_r = [
        ("Frozen Backbone: ", "Freezes pretrained weights W_0 in R^{d x k}; base weights require no optimizer moments or gradient tracking."),
        ("Low-Rank Parameterization: ", "Decomposes weight updates via low-rank matrices: ΔW = (α/r) · B · A, where B in R^{d x r} and A in R^{r x k} with rank r << min(d, k)."),
        ("Over 99% Parameter Reduction: ", "Trains < 1% of total parameters (~300K adapter params in RoBERTa vs 125M), drastically shrinking GPU VRAM."),
        ("Zero Deployment Overhead: ", "During deployment, adapter weights fold directly into base weights: W = W_0 + (α/r) · B · A with zero inference latency.")
    ]
    for i, (bold_txt, norm_txt) in enumerate(bullets_s1_r):
        p = tf1_r.paragraphs[0] if i == 0 else tf1_r.add_paragraph()
        p.space_after = Pt(10)
        r1 = p.add_run()
        r1.text = "• " + bold_txt
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = norm_txt
        r2.font.size = Pt(11.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s1, Inches(0.8), Inches(6.25), Inches(11.7), Inches(0.65), bg_color=CALLOUT_BG, border_color=CALLOUT_BORDER)
    tb1_b = s1.shapes.add_textbox(Inches(1.0), Inches(6.32), Inches(11.3), Inches(0.5))
    tf1_b = tb1_b.text_frame
    p1_b = tf1_b.paragraphs[0]
    r_b1 = p1_b.add_run()
    r_b1.text = "Core Takeaway: "
    r_b1.font.bold = True
    r_b1.font.size = Pt(11)
    r_b1.font.color.rgb = ACCENT_BLUE
    r_b2 = p1_b.add_run()
    r_b2.text = "PEFT overcomes deployment and memory bottlenecks. However, standard LoRA introduces distinct optimization dynamics compared to full fine-tuning, directly motivating Weight-Decomposed Low-Rank Adaptation (DoRA)."
    r_b2.font.size = Pt(11)
    r_b2.font.color.rgb = NAVY


    # =========================================================================
    # SLIDE 2: Existing Research: Weight-Decomposed Low-Rank Adaptation (DoRA)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(
        s2,
        "Part I: Existing Research & Problem Statement",
        "Existing Research: Weight-Decomposed Low-Rank Adaptation (DoRA)",
        "Decoupling directional updates from magnitude adjustments to mirror full fine-tuning dynamics (Liu et al., 2024)",
        2
    )

    add_card(s2, Inches(0.8), Inches(1.8), Inches(6.2), Inches(5.1), "Weight Decomposition Formulation")
    tb2_l = s2.shapes.add_textbox(Inches(1.0), Inches(2.35), Inches(5.8), Inches(4.3))
    tf2_l = tb2_l.text_frame
    tf2_l.word_wrap = True

    p = tf2_l.paragraphs[0]
    p.text = "Mathematical Formulation:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = NAVY

    eq_shape = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(2.7), Inches(5.8), Inches(0.75))
    eq_shape.fill.solid()
    eq_shape.fill.fore_color.rgb = DARK_BLUE
    eq_shape.line.fill.background()
    tf_eq = eq_shape.text_frame
    p_eq = tf_eq.paragraphs[0]
    p_eq.alignment = PP_ALIGN.CENTER
    r_eq = p_eq.add_run()
    r_eq.text = "W = m ⊙ ( V / ||V||_c ) = m ⊙ ( (W_0 + ΔW) / ||W_0 + ΔW||_c )"
    r_eq.font.bold = True
    r_eq.font.size = Pt(13.5)
    r_eq.font.color.rgb = WHITE

    tb2_eq_vars = s2.shapes.add_textbox(Inches(1.0), Inches(3.55), Inches(5.8), Inches(3.2))
    tf2_vars = tb2_eq_vars.text_frame
    tf2_vars.word_wrap = True
    eq_terms = [
        ("W ∈ R^{d x k}: ", "Adapted weight matrix reconstructed during the forward pass."),
        ("m ∈ R^{1 x k}: ", "Trainable magnitude vector initialized as ||W_0||_c, capturing channel-wise scale."),
        ("V = W_0 + ΔW: ", "Directional matrix combining frozen weights W_0 with low-rank update ΔW."),
        ("||V||_c: ", "Column-wise Euclidean norm, ensuring V / ||V||_c has normalized unit-length columns."),
        ("ΔW = (α/r) · B · A: ", "Low-rank directional adaptation with B initialized to 0 and A to Gaussian noise.")
    ]
    for i, (b_txt, n_txt) in enumerate(eq_terms):
        p = tf2_vars.paragraphs[0] if i == 0 else tf2_vars.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s2, Inches(7.3), Inches(1.8), Inches(5.2), Inches(5.1), "Decomposition Mechanism & Insight")
    tb2_r = s2.shapes.add_textbox(Inches(7.5), Inches(2.35), Inches(4.8), Inches(4.3))
    tf2_r = tb2_r.text_frame
    tf2_r.word_wrap = True

    insights = [
        ("Coupling Limitation in LoRA: ", "In conventional LoRA, ΔW alters both direction and magnitude simultaneously in a proportional manner, restricting optimization flexibility."),
        ("Closer to Full Fine-Tuning: ", "Liu et al. (2024) demonstrated that full fine-tuning exhibits subtle directional rotations paired with substantial magnitude adjustments. DoRA decouples these two pathways."),
        ("Directional Update (LoRA path): ", "Matrices B and A learn directional rotation ΔV without having to scale individual feature channels."),
        ("Magnitude Update (Vector m): ", "Vector m allows independent per-channel amplitude scaling with negligible parameter overhead (k parameters)."),
        ("Static Uniform Rank Baseline: ", "In standard published DoRA, every adapted layer receives an identical, fixed rank budget (e.g. r=3 across all attention and MLP projections).")
    ]
    for i, (b_txt, n_txt) in enumerate(insights):
        p = tf2_r.paragraphs[0] if i == 0 else tf2_r.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED


    # =========================================================================
    # SLIDE 3: Research Gap and Problem Statement
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(
        s3,
        "Part I: Existing Research & Problem Statement",
        "Research Gap and Problem Statement",
        "The limitation of static uniform rank allocation and pure rank pruning in low-rank adaptation",
        3
    )

    pillar_w = Inches(3.64)
    gap = Inches(0.39)

    add_card(s3, Inches(0.8), Inches(1.8), pillar_w, Inches(3.1), "1. Uniform Rank Allocation")
    tb3_p1 = s3.shapes.add_textbox(Inches(0.95), Inches(2.35), pillar_w - Inches(0.3), Inches(2.4))
    tf3_p1 = tb3_p1.text_frame
    tf3_p1.word_wrap = True
    p1_items = [
        ("Rigid Uniformity: ", "Every single projection across all 12-24 Transformer layers is assigned identical rank r (e.g. r=3)."),
        ("Functional Heterogeneity: ", "Transformer layers exhibit unequal representation depth. Lower layers capture general syntax; higher layers capture task semantics."),
        ("Capacity Inefficiency: ", "Uniform allocation over-allocates rank to low-demand layers while starving bottleneck layers.")
    ]
    for i, (b_txt, n_txt) in enumerate(p1_items):
        p = tf3_p1.paragraphs[0] if i == 0 else tf3_p1.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s3, Inches(0.8) + pillar_w + gap, Inches(1.8), pillar_w, Inches(3.1), "2. Pure Rank Pruning")
    tb3_p2 = s3.shapes.add_textbox(Inches(0.95) + pillar_w + gap, Inches(2.35), pillar_w - Inches(0.3), Inches(2.4))
    tf3_p2 = tb3_p2.text_frame
    tf3_p2.word_wrap = True
    p2_items = [
        ("Subtractive Mechanism: ", "Existing techniques (e.g. AdaLoRA, DoRA-pruning) score component utility and remove low-ranking slots."),
        ("Capacity Destruction: ", "Pruning is strictly one-way: capacity removed from unneeded components is discarded permanently from the active network."),
        ("No Reinvestment: ", "Layers that could achieve higher task accuracy with additional capacity cannot acquire the discarded rank.")
    ]
    for i, (b_txt, n_txt) in enumerate(p2_items):
        p = tf3_p2.paragraphs[0] if i == 0 else tf3_p2.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s3, Inches(0.8) + (pillar_w + gap)*2, Inches(1.8), pillar_w, Inches(3.1), "3. The Unaddressed Need")
    tb3_p3 = s3.shapes.add_textbox(Inches(0.95) + (pillar_w + gap)*2, Inches(2.35), pillar_w - Inches(0.3), Inches(2.4))
    tf3_p3 = tb3_p3.text_frame
    tf3_p3.word_wrap = True
    p3_items = [
        ("Dynamic Asymmetry: ", "Training gradients fluctuate; critical representations emerge dynamically rather than statically before training begins."),
        ("Budget Constraint: ", "Total parameter budget must remain fixed or bounded to satisfy deployment and inference requirements."),
        ("Missing Capability: ", "A formal, stable mechanism to dynamically transfer capacity from low-demand layers to high-demand layers.")
    ]
    for i, (b_txt, n_txt) in enumerate(p3_items):
        p = tf3_p3.paragraphs[0] if i == 0 else tf3_p3.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s3, Inches(0.8), Inches(5.15), Inches(11.7), Inches(1.75), bg_color=DARK_BLUE, border_color=None)
    tb3_call = s3.shapes.add_textbox(Inches(1.0), Inches(5.25), Inches(11.3), Inches(1.55))
    tf3_call = tb3_call.text_frame
    tf3_call.word_wrap = True

    p_cq = tf3_call.paragraphs[0]
    p_cq.space_after = Pt(4)
    r_cq_lbl = p_cq.add_run()
    r_cq_lbl.text = "CORE RESEARCH QUESTION:  "
    r_cq_lbl.font.bold = True
    r_cq_lbl.font.size = Pt(12)
    r_cq_lbl.font.color.rgb = ACCENT_AMBER
    r_cq = p_cq.add_run()
    r_cq.text = "How can a fixed or dynamically constrained low-rank parameter budget be allocated across Transformer layers according to their adaptation requirements, rather than relying only on uniform allocation or rank removal?"
    r_cq.font.size = Pt(12)
    r_cq.font.bold = True
    r_cq.font.color.rgb = WHITE

    p_ro = tf3_call.add_paragraph()
    r_ro_lbl = p_ro.add_run()
    r_ro_lbl.text = "PROPOSED OBJECTIVE:  "
    r_ro_lbl.font.bold = True
    r_ro_lbl.font.size = Pt(12)
    r_ro_lbl.font.color.rgb = ACCENT_TEAL
    r_ro = p_ro.add_run()
    r_ro.text = "To propose a budget-constrained dynamic rank reallocation framework that transfers low-rank adaptation capacity from less useful layers to layers with greater estimated adaptation demand."
    r_ro.font.size = Pt(12)
    r_ro.font.color.rgb = RGBColor(226, 232, 240)


    # =========================================================================
    # SLIDE 4: Proposed Framework: Dynamic Rank Reallocation
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(
        s4,
        "Part II: Proposed Dynamic Rank Reallocation Extension",
        "Proposed Framework: Dynamic Rank Reallocation",
        "A budget-preserving capacity transfer strategy across Transformer layers during training",
        4
    )

    add_card(s4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.3), "Core Philosophy: Reallocation vs. Pruning")
    tb4_l = s4.shapes.add_textbox(Inches(1.0), Inches(2.35), Inches(5.2), Inches(3.6))
    tf4_l = tb4_l.text_frame
    tf4_l.word_wrap = True

    philo_points = [
        ("Beyond Subtractive Pruning: ", "Traditional dynamic pruning only decreases rank. The proposed method introduces capacity redistribution, keeping the total active rank budget strictly conserved."),
        ("Donor vs. Recipient Concept: ", "Layers exhibiting low utility become eligible Donors to surrender 1 active rank component. Layers with high adaptation demand become eligible Recipients to receive 1 rank component."),
        ("Budget Conservation Invariant: ", "Every reallocation event preserves the global rank budget B_t exactly: losing 1 rank in the donor while adding 1 rank in the recipient guarantees ΔB_t = 0."),
        ("Autonomous Specialization: ", "Enables the network to dynamically discover layer-wise importance patterns without human intervention or exhaustive hyperparameter sweeps.")
    ]
    for i, (b_txt, n_txt) in enumerate(philo_points):
        p = tf4_l.paragraphs[0] if i == 0 else tf4_l.add_paragraph()
        p.space_after = Pt(10)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = norm_txt
        r2.font.size = Pt(11.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s4, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.3), "Illustrative Allocation Transfer Example")
    
    add_card(s4, Inches(7.1), Inches(2.4), Inches(5.2), Inches(1.5), bg_color=RGBColor(241, 245, 249), border_color=CARD_BORDER)
    tb_b4 = s4.shapes.add_textbox(Inches(7.2), Inches(2.45), Inches(5.0), Inches(1.4))
    tf_b4 = tb_b4.text_frame
    tf_b4.word_wrap = True
    p = tf_b4.paragraphs[0]
    p.text = "INITIAL ALLOCATION STATE (Global Budget B = 6)"
    p.font.bold = True
    p.font.size = Pt(10)
    p.font.color.rgb = ACCENT_BLUE
    
    p = tf_b4.add_paragraph()
    p.text = "• Layer 1 (Shallow MLP): Rank = 3  [Low Adaptation Utility]"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN
    p = tf_b4.add_paragraph()
    p.text = "• Layer 2 (Mid Attention): Rank = 2  [Moderate Utility]"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN
    p = tf_b4.add_paragraph()
    p.text = "• Layer 3 (Deep Attention): Rank = 1  [High Gradient Demand]"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    arr_box = s4.shapes.add_textbox(Inches(7.1), Inches(3.95), Inches(5.2), Inches(0.35))
    tf_arr = arr_box.text_frame
    p_arr = tf_arr.paragraphs[0]
    p_arr.alignment = PP_ALIGN.CENTER
    r_arr = p_arr.add_run()
    r_arr.text = "▼  Reallocation Event: Transfer 1 Rank from Layer 1 to Layer 3  ▼"
    r_arr.font.bold = True
    r_arr.font.size = Pt(10.5)
    r_arr.font.color.rgb = ACCENT_AMBER

    add_card(s4, Inches(7.1), Inches(4.35), Inches(5.2), Inches(1.5), bg_color=CALLOUT_BG, border_color=CALLOUT_BORDER)
    tb_a4 = s4.shapes.add_textbox(Inches(7.2), Inches(4.4), Inches(5.0), Inches(1.4))
    tf_a4 = tb_a4.text_frame
    tf_a4.word_wrap = True
    p = tf_a4.paragraphs[0]
    p.text = "REALLOCATED ALLOCATION STATE (Global Budget B = 6 Conserved)"
    p.font.bold = True
    p.font.size = Pt(10)
    p.font.color.rgb = ACCENT_TEAL
    
    p = tf_a4.add_paragraph()
    p.text = "• Layer 1 (Donor): Rank = 2  (Decreased from 3 to 2)"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN
    p = tf_a4.add_paragraph()
    p.text = "• Layer 2 (Neutral): Rank = 2  (Unchanged)"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN
    p = tf_a4.add_paragraph()
    p.text = "• Layer 3 (Recipient): Rank = 2  (Increased from 1 to 2)"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    callout4 = add_card(s4, Inches(0.8), Inches(6.25), Inches(11.7), Inches(0.65), bg_color=CALLOUT_BG, border_color=CALLOUT_BORDER)
    tb4_b = s4.shapes.add_textbox(Inches(1.0), Inches(6.32), Inches(11.3), Inches(0.5))
    tf4_b = tb4_b.text_frame
    p4_b = tf4_b.paragraphs[0]
    r_b1 = p4_b.add_run()
    r_b1.text = "Methodological Clarification: "
    r_b1.font.bold = True
    r_b1.font.size = Pt(11)
    r_b1.font.color.rgb = ACCENT_BLUE
    r_b2 = p4_b.add_run()
    r_b2.text = "This toy example is purely illustrative to explain the redistribution principle. The proposed framework targets 72 adaptive layers in modern Transformer backbones under formal stability and budget constraints."
    r_b2.font.size = Pt(11)
    r_b2.font.color.rgb = NAVY


    # =========================================================================
    # SLIDE 5: Measuring Adaptation Requirements
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(
        s5,
        "Part II: Proposed Dynamic Rank Reallocation Extension",
        "Measuring Adaptation Requirements",
        "A dual-signal formulation combining retrospective component utility with prospective gradient demand",
        5
    )

    add_card(s5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.3), "Signal 1: Component Importance (D_l)")
    
    eq1_shape = s5.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(2.4), Inches(5.2), Inches(0.65))
    eq1_shape.fill.solid()
    eq1_shape.fill.fore_color.rgb = DARK_BLUE
    eq1_shape.line.fill.background()
    tf_eq1 = eq1_shape.text_frame
    p_eq1 = tf_eq1.paragraphs[0]
    p_eq1.alignment = PP_ALIGN.CENTER
    r_eq1 = p_eq1.add_run()
    r_eq1.text = "D_l = ( 1 / r_l ) · Σ_{j ∈ A_l} I_{l,j}"
    r_eq1.font.bold = True
    r_eq1.font.size = Pt(14)
    r_eq1.font.color.rgb = WHITE

    tb5_l = s5.shapes.add_textbox(Inches(1.0), Inches(3.2), Inches(5.2), Inches(2.8))
    tf5_l = tb5_l.text_frame
    tf5_l.word_wrap = True
    d_points = [
        ("Retrospective Utility Signal: ", "Quantifies how useful the currently active low-rank components in layer l have been during recent forward passes."),
        ("Frobenius Contribution (I_{l,j}): ", "Derived from the relative Frobenius norm contribution of each individual low-rank component j to the directional update matrix ΔW."),
        ("Active Set Normalization (1/r_l): ", "Averaging over the set of active components A_l normalizes against layer rank, allowing equitable comparison across layers with different ranks."),
        ("Limitation if Used Alone: ", "Only measures existing components; cannot estimate whether a layer would benefit from having additional rank capacity.")
    ]
    for i, (b_txt, n_txt) in enumerate(d_points):
        p = tf5_l.paragraphs[0] if i == 0 else tf5_l.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s5, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.3), "Signal 2: Gradient Demand (G_l)")

    eq2_shape = s5.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.1), Inches(2.4), Inches(5.2), Inches(0.65))
    eq2_shape.fill.solid()
    eq2_shape.fill.fore_color.rgb = DARK_BLUE
    eq2_shape.line.fill.background()
    tf_eq2 = eq2_shape.text_frame
    p_eq2 = tf_eq2.paragraphs[0]
    p_eq2.alignment = PP_ALIGN.CENTER
    r_eq2 = p_eq2.add_run()
    r_eq2.text = "G_l = ( ||∇A_l||_F / √|A_l| ) + ( ||∇B_l||_F / √|B_l| )"
    r_eq2.font.bold = True
    r_eq2.font.size = Pt(13)
    r_eq2.font.color.rgb = WHITE

    tb5_r = s5.shapes.add_textbox(Inches(7.1), Inches(3.2), Inches(5.2), Inches(2.8))
    tf5_r = tb5_r.text_frame
    tf5_r.word_wrap = True
    g_points = [
        ("Prospective Sensitivity Signal: ", "Measures how forcefully the adapter matrices A_l and B_l are being pulled by task gradients during backpropagation."),
        ("Dimensional Normalization: ", "Scaling by square root of parameter count (√|A_l|, √|B_l|) prevents larger projection matrices from dominating solely due to higher dimensionality."),
        ("Demand Indicator: ", "Layers experiencing strong, sustained gradient pressure signal an urgency for additional representational degrees of freedom."),
        ("Cautionary Interpretation: ", "High gradient magnitude does not definitively prove a layer needs rank—it may indicate transient training noise, requiring smoothing.")
    ]
    for i, (b_txt, n_txt) in enumerate(g_points):
        p = tf5_r.paragraphs[0] if i == 0 else tf5_r.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED

    callout5 = add_card(s5, Inches(0.8), Inches(6.25), Inches(11.7), Inches(0.65), bg_color=CALLOUT_BG, border_color=CALLOUT_BORDER)
    tb5_b = s5.shapes.add_textbox(Inches(1.0), Inches(6.32), Inches(11.3), Inches(0.5))
    tf5_b = tb5_b.text_frame
    p5_b = tf5_b.paragraphs[0]
    r_b1 = p5_b.add_run()
    r_b1.text = "Dual-Signal Synthesis: "
    r_b1.font.bold = True
    r_b1.font.size = Pt(11)
    r_b1.font.color.rgb = ACCENT_BLUE
    r_b2 = p5_b.add_run()
    r_b2.text = "Importance captures retrospective utility of existing components; Gradient Demand captures prospective capacity hunger. Combining both prevents premature rank reallocations based on one incomplete metric."
    r_b2.font.size = Pt(11)
    r_b2.font.color.rgb = NAVY


    # =========================================================================
    # SLIDE 6: Utility Estimation and Parameter-Aware Allocation
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(
        s6,
        "Part II: Proposed Dynamic Rank Reallocation Extension",
        "Utility Estimation and Parameter-Aware Allocation",
        "Smoothing signals over time, combining utilities, and prioritizing layers by adaptation value per parameter",
        6
    )

    card_w = Inches(3.64)
    gap = Inches(0.39)

    add_card(s6, Inches(0.8), Inches(1.8), card_w, Inches(4.3), "1. EMA Temporal Smoothing")
    tb6_c1 = s6.shapes.add_textbox(Inches(0.95), Inches(2.35), card_w - Inches(0.3), Inches(3.6))
    tf6_c1 = tb6_c1.text_frame
    tf6_c1.word_wrap = True

    c1_content = [
        ("Formulations: ", "Smooth both signals across optimizer steps t:\n• D̄_l^t = β · D̄_l^{t-1} + (1-β) · D_l^t\n• Ḡ_l^t = β · Ḡ_l^{t-1} + (1-β) · G_l^t"),
        ("Smoothing Parameter (β = 0.9): ", "Weights recent historical trend heavily while incorporating current batch evidence."),
        ("Eliminating Churn: ", "Mini-batch gradients fluctuate wildly. Smoothing prevents erratic rank thrashing caused by outlier batches."),
        ("Stable Trendline: ", "Provides a reliable baseline for making architectural adjustments.")
    ]
    for i, (b_txt, n_txt) in enumerate(c1_content):
        p = tf6_c1.paragraphs[0] if i == 0 else tf6_c1.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s6, Inches(0.8) + card_w + gap, Inches(1.8), card_w, Inches(4.3), "2. Combined Layer Utility")
    tb6_c2 = s6.shapes.add_textbox(Inches(0.95) + card_w + gap, Inches(2.35), card_w - Inches(0.3), Inches(3.6))
    tf6_c2 = tb6_c2.text_frame
    tf6_c2.word_wrap = True

    c2_content = [
        ("Layer Utility Formula: ", "U_l = λ · Norm(Ḡ_l) + (1-λ) · Norm(D̄_l)"),
        ("Equal Initial Weight (λ = 0.5): ", "Gives balanced 50/50 weighting between prospective gradient demand and retrospective importance."),
        ("Layer-Wise Normalization: ", "Norm(·) applies min-max scaling across all eligible layers in the network, bringing both signals into the comparable range [0, 1]."),
        ("Single Comparison Metric: ", "Produces a unified utility score U_l ranking all Transformer layers from lowest utility to highest adaptation need.")
    ]
    for i, (b_txt, n_txt) in enumerate(c2_content):
        p = tf6_c2.paragraphs[0] if i == 0 else tf6_c2.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s6, Inches(0.8) + (card_w + gap)*2, Inches(1.8), card_w, Inches(4.3), "3. Cost-Adjusted Utility (Q_l)")
    tb6_c3 = s6.shapes.add_textbox(Inches(0.95) + (card_w + gap)*2, Inches(2.35), card_w - Inches(0.3), Inches(3.6))
    tf6_c3 = tb6_c3.text_frame
    tf6_c3.word_wrap = True

    c3_content = [
        ("Parameter Cost Formulation: ", "C_l = d_{in,l} + d_{out,l} + 1\nAccounts for input/output dimensions plus gating scalar."),
        ("Cost-Adjusted Metric: ", "Q_l = U_l / C_l\nMeasures adaptation utility per added parameter."),
        ("Dimensional Asymmetry: ", "In RoBERTa, MLP intermediate projections (768 x 3072) cost 3,841 params/rank. Attention projections (768 x 768) cost 1,537 params/rank (2.5x cheaper)."),
        ("Efficiency Prioritization: ", "A layer with high raw utility may be an inefficient recipient if its parameter cost is massive.")
    ]
    for i, (b_txt, n_txt) in enumerate(c3_content):
        p = tf6_c3.paragraphs[0] if i == 0 else tf6_c3.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED

    callout6 = add_card(s6, Inches(0.8), Inches(6.25), Inches(11.7), Inches(0.65), bg_color=CALLOUT_BG, border_color=CALLOUT_BORDER)
    tb6_b = s6.shapes.add_textbox(Inches(1.0), Inches(6.32), Inches(11.3), Inches(0.5))
    tf6_b = tb6_b.text_frame
    p6_b = tf6_b.paragraphs[0]
    r_b1 = p6_b.add_run()
    r_b1.text = "Parameter Efficiency Insight: "
    r_b1.font.bold = True
    r_b1.font.size = Pt(11)
    r_b1.font.color.rgb = ACCENT_BLUE
    r_b2 = p6_b.add_run()
    r_b2.text = "Raw utility identifies where adaptation is wanted; Cost-adjusted utility Q_l ensures capacity is routed where it delivers maximum impact per parameter, preventing parameter bloat in wide MLP layers."
    r_b2.font.size = Pt(11)
    r_b2.font.color.rgb = NAVY


    # =========================================================================
    # SLIDE 7: Dynamic Rank Reallocation Algorithm
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(
        s7,
        "Part II: Proposed Dynamic Rank Reallocation Extension",
        "Dynamic Rank Reallocation Algorithm",
        "Complete procedural execution cycle integrating evaluation, candidate selection, stability checks, and transfer",
        7
    )

    add_card(s7, Inches(0.8), Inches(1.8), Inches(7.5), Inches(5.1), "Reallocation Procedure Execution Cycle")

    phases = [
        ("Phase 1: Metrics & Smoothing (Step % 100 == 0)",
         "Compute component importance D_l and gradient demand G_l across all layers.\nUpdate temporal EMAs: D̄_l^t = 0.9·D̄_l^{t-1} + 0.1·D_l,  Ḡ_l^t = 0.9·Ḡ_l^{t-1} + 0.1·G_l."),
        ("Phase 2: Layer Utility & Cost Calculation",
         "Calculate normalized combined utility U_l = 0.5·Norm(Ḡ_l) + 0.5·Norm(D̄_l).\nCompute parameter cost C_l = d_in + d_out + 1 and cost-adjusted score Q_l = U_l / C_l."),
        ("Phase 3: Donor & Recipient Candidate Selection",
         "Donor Candidate: Layer with lowest utility U_l satisfying r_l > r_min (1) and cooldown > 500.\nRecipient Candidate: Layer with highest cost-adjusted Q_l satisfying r_l < r_max (4) and cooldown > 500."),
        ("Phase 4: Guard Verification & Atomic Rank Transfer",
         "Verify transfer margin: U_{rec} - U_{don} > 0.1 (10% utility gap).\nDeactivate least important component in Donor; activate initialized slot in Recipient.\nReset optimizer moments for new component; verify budget invariant Σ r_l = B_t.")
    ]

    top_offset = 2.4
    for i, (title_p, body_p) in enumerate(phases):
        p_card = add_card(s7, Inches(1.0), Inches(top_offset), Inches(7.1), Inches(0.98), bg_color=RGBColor(248, 250, 252), border_color=CARD_BORDER)
        tb_p = s7.shapes.add_textbox(Inches(1.15), Inches(top_offset + 0.05), Inches(6.8), Inches(0.88))
        tf_p = tb_p.text_frame
        tf_p.word_wrap = True
        
        p = tf_p.paragraphs[0]
        p.text = title_p
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = ACCENT_BLUE if i < 2 else (ACCENT_AMBER if i == 2 else ACCENT_TEAL)
        
        p_b = tf_p.add_paragraph()
        p_b.text = body_p
        p_b.font.size = Pt(10)
        p_b.font.color.rgb = TEXT_MUTED
        top_offset += 1.08

    add_card(s7, Inches(8.5), Inches(1.8), Inches(4.0), Inches(5.1), "Candidate Selection Rules")
    tb7_r = s7.shapes.add_textbox(Inches(8.7), Inches(2.35), Inches(3.6), Inches(4.3))
    tf7_r = tb7_r.text_frame
    tf7_r.word_wrap = True

    rules = [
        ("Donor Eligibility: ", "• Must maintain active rank r_l > r_min (≥ 1)\n• Must not have participated in a transfer within past 500 steps\n• Possesses minimum utility U_l across all eligible layers"),
        ("Recipient Eligibility: ", "• Must have spare physical slot r_l < r_max (≤ 4)\n• Must satisfy 500-step cooldown\n• Highest cost-adjusted utility Q_l = U_l / C_l"),
        ("Non-Direct Copying: ", "Crucial: Donor tensors are NOT copied to recipient due to dimension mismatch. Recipient slot is freshly initialized and trained."),
        ("Atomic Conservation: ", "Donor rank is decremented simultaneously as Recipient rank is incremented, guaranteeing zero budget deviation.")
    ]
    for i, (b_txt, n_txt) in enumerate(rules):
        p = tf7_r.paragraphs[0] if i == 0 else tf7_r.add_paragraph()
        p.space_after = Pt(10)
        r1 = p.add_run()
        r1.text = b_txt + "\n"
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10)
        r2.font.color.rgb = TEXT_MUTED


    # =========================================================================
    # SLIDE 8: Rank Budget, Constraints and Stability
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(
        s8,
        "Part II: Proposed Dynamic Rank Reallocation Extension",
        "Rank Budget, Constraints and Stability",
        "Rigorous budget preservation, hyperparameter boundaries, and structural oscillation safeguards",
        8
    )

    add_card(s8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.3), "Mathematical Budget Invariant")
    
    inv_shape = s8.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(2.4), Inches(5.2), Inches(0.65))
    inv_shape.fill.solid()
    inv_shape.fill.fore_color.rgb = DARK_BLUE
    inv_shape.line.fill.background()
    tf_inv = inv_shape.text_frame
    p_inv = tf_inv.paragraphs[0]
    p_inv.alignment = PP_ALIGN.CENTER
    r_inv = p_inv.add_run()
    r_inv.text = "Σ_{l=1}^L r_l^{(t)} = B_t"
    r_inv.font.bold = True
    r_inv.font.size = Pt(16)
    r_inv.font.color.rgb = WHITE

    tb8_l = s8.shapes.add_textbox(Inches(1.0), Inches(3.2), Inches(5.2), Inches(2.8))
    tf8_l = tb8_l.text_frame
    tf8_l.word_wrap = True
    inv_points = [
        ("Global Budget Invariant: ", "At any training step t, the sum of all active layer ranks exactly equals the target budget B_t."),
        ("Decoupling from Scheduled Pruning: ", "If a global pruning schedule decays total rank (e.g. from 216 to 144 across steps 4,738 to 15,795), B_t decreases along that scheduled curve. A reallocation event never alters B_t."),
        ("Strict Conservation: ", "Δr_recipient = +1 and Δr_donor = -1 ensures Δ(Σ r_l) = 0 at every reallocation step."),
        ("No Capacity Drift: ", "Guarantees parameter footprint at inference matches pre-specified deployment constraints.")
    ]
    for i, (b_txt, n_txt) in enumerate(inv_points):
        p = tf8_l.paragraphs[0] if i == 0 else tf8_l.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s8, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.3), "Proposed Hyperparameters & Stability Safeguards")
    
    rows, cols = 6, 2
    table_shape = s8.shapes.add_table(rows, cols, Inches(7.1), Inches(2.4), Inches(5.2), Inches(2.0))
    table = table_shape.table
    table.columns[0].width = Inches(3.4)
    table.columns[1].width = Inches(1.8)

    table_data = [
        ("Initial Rank per Layer (r_0)", "3  (Total 216)"),
        ("Minimum Active Rank (r_min)", "1  (No layer death)"),
        ("Maximum Capacity per Layer (r_max)", "4  (Headroom for recipient)"),
        ("Reallocation Interval (k)", "100 optimizer steps"),
        ("Transfer Cooldown Period", "500 optimizer steps"),
        ("Transfer Margin Threshold (ΔU)", "0.1  (10% utility gap)")
    ]
    for r_idx, (k, v) in enumerate(table_data):
        cell_k = table.cell(r_idx, 0)
        cell_v = table.cell(r_idx, 1)
        cell_k.text = k
        cell_v.text = v
        for c in [cell_k, cell_v]:
            p = c.text_frame.paragraphs[0]
            p.font.size = Pt(9.5)
            p.font.name = "Calibri"
            if r_idx == 0:
                p.font.bold = True

    tb8_r = s8.shapes.add_textbox(Inches(7.1), Inches(4.5), Inches(5.2), Inches(1.5))
    tf8_r = tb8_r.text_frame
    tf8_r.word_wrap = True

    safeguards = [
        ("Transfer Margin Guard: ", "Transfer occurs only if U_{rec} - U_{don} > 0.1, preventing oscillation over negligible utility differences."),
        ("Cooldown Constraint: ", "Prevents 'ping-pong' instability where rank components rapidly bounce between identical layers."),
        ("Single Transfer Cap: ", "At most 1 rank transferred per event to maintain optimization smoothness.")
    ]
    for i, (b_txt, n_txt) in enumerate(safeguards):
        p = tf8_r.paragraphs[0] if i == 0 else tf8_r.add_paragraph()
        p.space_after = Pt(4)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = TEXT_MUTED

    callout8 = add_card(s8, Inches(0.8), Inches(6.25), Inches(11.7), Inches(0.65), bg_color=CALLOUT_BG, border_color=CALLOUT_BORDER)
    tb8_b = s8.shapes.add_textbox(Inches(1.0), Inches(6.32), Inches(11.3), Inches(0.5))
    tf8_b = tb8_b.text_frame
    p8_b = tf8_b.paragraphs[0]
    r_b1 = p8_b.add_run()
    r_b1.text = "Architectural Stability Principle: "
    r_b1.font.bold = True
    r_b1.font.size = Pt(11)
    r_b1.font.color.rgb = ACCENT_BLUE
    r_b2 = p8_b.add_run()
    r_b2.text = "Dynamic adaptation must never sacrifice optimization stability. Strict capacity bounds [1, 4], margin thresholds (0.1), and cooldowns ensure controlled, gradual capacity shifts without gradient spikes."
    r_b2.font.size = Pt(11)
    r_b2.font.color.rgb = NAVY


    # =========================================================================
    # SLIDE 9: Proposed Experimental Evaluation
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(
        s9,
        "Part II: Proposed Dynamic Rank Reallocation Extension",
        "Proposed Experimental Evaluation",
        "A rigorous empirical evaluation protocol designed to test adaptation efficiency against matched baselines",
        9
    )

    add_card(s9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.3), "Planned Benchmark Protocol")
    tb9_l = s9.shapes.add_textbox(Inches(1.0), Inches(2.35), Inches(5.2), Inches(3.6))
    tf9_l = tb9_l.text_frame
    tf9_l.word_wrap = True

    proto_points = [
        ("Base Model Architecture: ", "roberta-base (12 encoder layers, 72 linear projections adapted: W_q, W_k, W_v, W_o, W_f1, W_f2)."),
        ("Target Benchmark Dataset: ", "GLUE Stanford Sentiment Treebank (SST-2: 67,349 train / 872 validation samples)."),
        ("Training Duration & Setup: ", "30 epochs, batch size 64, AdamW optimizer, learning rate 8e-4, seed 42 pilot."),
        ("Global Rank Constraint: ", "Initial rank 3 (216 total), target budget 144 (average rank 2.0), max layer capacity 4."),
        ("Strict Experimental Hygiene: ", "All comparisons share identical seeds, learning rates, batch sizes, and global rank budgets.")
    ]
    for i, (b_txt, n_txt) in enumerate(proto_points):
        p = tf9_l.paragraphs[0] if i == 0 else tf9_l.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s9, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.3), "Matched Comparison Groups & Evaluation Metrics")
    tb9_r = s9.shapes.add_textbox(Inches(7.1), Inches(2.35), Inches(5.2), Inches(3.6))
    tf9_r = tb9_r.text_frame
    tf9_r.word_wrap = True

    comp_groups = [
        ("Group 1 — Uniform Static Rank: ", "Standard DoRA with uniform rank r=2 fixed across all 72 layers (Budget = 144)."),
        ("Group 2 — Dynamic Pruning: ", "DoRA with cubic rank decay (216 → 144) without reallocation (existing baseline)."),
        ("Group 3 — Proposed Dynamic Reallocation: ", "Importance + Gradient Demand + Parameter Cost reallocation within budget 144."),
        ("Group 4 — Random Reallocation (Control): ", "Random donor/recipient transfers under identical budget and constraints.")
    ]
    for i, (b_txt, n_txt) in enumerate(comp_groups):
        p = tf9_r.paragraphs[0] if i == 0 else tf9_r.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = b_txt
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = ACCENT_BLUE if i == 2 else NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10)
        r2.font.color.rgb = TEXT_MUTED

    p_m = tf9_r.add_paragraph()
    p_m.space_before = Pt(8)
    r_m_lbl = p_m.add_run()
    r_m_lbl.text = "Planned Evaluation Metrics: "
    r_m_lbl.font.bold = True
    r_m_lbl.font.size = Pt(10.5)
    r_m_lbl.font.color.rgb = ACCENT_TEAL
    r_m_txt = p_m.add_run()
    r_m_txt.text = "\n• Accuracy, Macro-F1, Weighted-F1\n• Final layer-wise active rank distribution\n• Reallocation event count, training runtime overhead\n• Adapter physical parameter count & storage size (MB)\n• Inference latency (ms) & peak GPU VRAM (MB)"
    r_m_txt.font.size = Pt(10)
    r_m_txt.font.color.rgb = TEXT_MUTED

    callout9 = add_card(s9, Inches(0.8), Inches(6.25), Inches(11.7), Inches(0.65), bg_color=CALLOUT_BG, border_color=CALLOUT_BORDER)
    tb9_b = s9.shapes.add_textbox(Inches(1.0), Inches(6.32), Inches(11.3), Inches(0.5))
    tf9_b = tb9_b.text_frame
    p9_b = tf9_b.paragraphs[0]
    r_b1 = p9_b.add_run()
    r_b1.text = "Scientific Integrity Notice: "
    r_b1.font.bold = True
    r_b1.font.size = Pt(11)
    r_b1.font.color.rgb = ACCENT_ROSE
    r_b2 = p9_b.add_run()
    r_b2.text = "This protocol outlines a planned research evaluation. The proposed framework has not yet been experimentally executed; no performance numbers or superiority claims are fabricated."
    r_b2.font.size = Pt(11)
    r_b2.font.color.rgb = NAVY


    # =========================================================================
    # SLIDE 10: Expected Contributions, Limitations and Conclusion
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(
        s10,
        "Part II: Proposed Dynamic Rank Reallocation Extension",
        "Expected Contributions, Limitations and Conclusion",
        "Synthesizing theoretical advances, scientific hypotheses, critical limitations, and research outlook",
        10
    )

    col_w = Inches(3.64)
    gap = Inches(0.39)

    add_card(s10, Inches(0.8), Inches(1.8), col_w, Inches(4.3), "Core Research Contributions")
    tb10_c1 = s10.shapes.add_textbox(Inches(0.95), Inches(2.35), col_w - Inches(0.3), Inches(3.6))
    tf10_c1 = tb10_c1.text_frame
    tf10_c1.word_wrap = True

    c1_pts = [
        ("Dynamic Redistribution: ", "Proposes dynamic rank redistribution rather than static allocation or subtractive pruning."),
        ("Dual-Signal Measurement: ", "Synthesizes retrospective component utility D_l with prospective gradient demand G_l."),
        ("Parameter Cost Awareness: ", "Normalizes capacity transfer by physical parameter cost (Q_l = U_l / C_l)."),
        ("Strict Budget Conservation: ", "Preserves global parameter footprint exactly via invariant Σ r_l = B_t.")
    ]
    for i, (b_txt, n_txt) in enumerate(c1_pts):
        p = tf10_c1.paragraphs[0] if i == 0 else tf10_c1.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s10, Inches(0.8) + col_w + gap, Inches(1.8), col_w, Inches(4.3), "Scientific Hypotheses")
    tb10_c2 = s10.shapes.add_textbox(Inches(0.95) + col_w + gap, Inches(2.35), col_w - Inches(0.3), Inches(3.6))
    tf10_c2 = tb10_c2.text_frame
    tf10_c2.word_wrap = True

    c2_pts = [
        ("Superior Capacity Allocation: ", "Hypothesizes that concentrating rank in high-demand layers yields higher task accuracy than uniform baselines."),
        ("Automated Layer Specialization: ", "Hypothesizes that distinct layer patterns (e.g. deeper attention layers) will autonomously emerge as key recipients."),
        ("Parameter-Efficiency Pareto: ", "Hypothesizes an improved trade-off between active parameter count and downstream task accuracy.")
    ]
    for i, (b_txt, n_txt) in enumerate(c2_pts):
        p = tf10_c2.paragraphs[0] if i == 0 else tf10_c2.add_paragraph()
        p.space_after = Pt(10)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s10, Inches(0.8) + (col_w + gap)*2, Inches(1.8), col_w, Inches(4.3), "Current Limitations & Scope")
    tb10_c3 = s10.shapes.add_textbox(Inches(0.95) + (col_w + gap)*2, Inches(2.35), col_w - Inches(0.3), Inches(3.6))
    tf10_c3 = tb10_c3.text_frame
    tf10_c3.word_wrap = True

    c3_pts = [
        ("Unvalidated Proposal: ", "The framework is currently a theoretical design; rigorous empirical testing is required before claiming benefits."),
        ("Computational Overhead: ", "Periodic gradient norm tracking and EMA updates introduce additional training runtime complexity."),
        ("Hyperparameter Sensitivity: ", "Sensitivity to transfer margin (0.1), cooldown (500), and utility weighting (λ) must be ablated.")
    ]
    for i, (b_txt, n_txt) in enumerate(c3_pts):
        p = tf10_c3.paragraphs[0] if i == 0 else tf10_c3.add_paragraph()
        p.space_after = Pt(10)
        r1 = p.add_run()
        r1.text = "• " + b_txt
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = n_txt
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = TEXT_MUTED

    add_card(s10, Inches(0.8), Inches(6.25), Inches(11.7), Inches(0.65), bg_color=DARK_BLUE, border_color=None)
    tb10_b = s10.shapes.add_textbox(Inches(1.0), Inches(6.32), Inches(11.3), Inches(0.5))
    tf10_b = tb10_b.text_frame
    p10_b = tf10_b.paragraphs[0]
    r_b1 = p10_b.add_run()
    r_b1.text = "Concluding Summary: "
    r_b1.font.bold = True
    r_b1.font.size = Pt(11)
    r_b1.font.color.rgb = ACCENT_AMBER
    r_b2 = p10_b.add_run()
    r_b2.text = "By reformulating low-rank adaptation from a static uniform budget into a dynamic, budget-conserving reallocation framework, this research establishes a principled mathematical and algorithmic foundation for parameter-efficient adaptation."
    r_b2.font.size = Pt(11)
    r_b2.font.color.rgb = WHITE

    pptx_path = out_dir / "AI_DoRA_Research_Presentation.pptx"
    prs.save(pptx_path)
    print(f"Presentation saved to {pptx_path} (Slide count: {len(prs.slides)})")


# =============================================================================
# WORD SCRIPT GENERATOR (Strictly 45-75s per slide, ~130-165 words)
# =============================================================================

def create_script():
    doc = docx.Document()

    for s in doc.sections:
        s.top_margin = DocxInches(1.0)
        s.bottom_margin = DocxInches(1.0)
        s.left_margin = DocxInches(1.0)
        s.right_margin = DocxInches(1.0)

    title_p = doc.add_paragraph()
    r_title = title_p.add_run("AI_DoRA: Dynamic Rank Reallocation Framework")
    r_title.font.name = "Arial"
    r_title.font.size = DocxPt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = DocxRGBColor(15, 23, 42)

    sub_p = doc.add_paragraph()
    r_sub = sub_p.add_run("Slide-by-Slide Complete Academic Speaking Script (10 Slides)")
    r_sub.font.name = "Calibri"
    r_sub.font.size = DocxPt(13)
    r_sub.font.color.rgb = DocxRGBColor(71, 85, 105)
    sub_p.paragraph_format.space_after = DocxPt(16)

    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = DocxPt(20)
    r_meta_lbl = meta_p.add_run("Presentation Delivery Guidelines: ")
    r_meta_lbl.font.bold = True
    r_meta_lbl.font.size = DocxPt(10.5)
    r_meta_lbl.font.color.rgb = DocxRGBColor(2, 132, 199)
    r_meta = meta_p.add_run(
        "Each section below corresponds to exactly one slide in AI_DoRA_Research_Presentation.pptx. "
        "The script is written in first-person academic presentation style, containing the exact spoken words, "
        "verbal explanations of mathematical equations, systematic diagram walkthroughs, and natural transitions. "
        "Every slide is strictly calibrated to 45 to 75 seconds of speaking time (~130 to 165 words)."
    )
    r_meta.font.size = DocxPt(10)
    r_meta.font.color.rgb = DocxRGBColor(71, 85, 105)

    scripts = [
        # SLIDE 1
        (
            1,
            "Background: Parameter-Efficient Fine-Tuning",
            "55-60 seconds",
            "Good morning. Today, I am presenting our research on parameter-efficient adaptation of foundation models, focusing on Weight-Decomposed Low-Rank Adaptation and our proposed framework for Dynamic Rank Reallocation.\n\n"
            "As shown on the left, adapting massive pretrained models like RoBERTa through full fine-tuning presents severe practical bottlenecks. Updating all parameters requires optimizers like AdamW to maintain first and second gradient moments for every weight, multiplying GPU VRAM requirements four to six times. Furthermore, serving dozens of downstream tasks requires storing independent multi-gigabyte checkpoints for each task, leading to deployment storage explosion.\n\n"
            "To resolve this, Parameter-Efficient Fine-Tuning, or PEFT, freezes the pretrained base weights completely. Low-Rank Adaptation, or LoRA, parameterizes updates via low-rank matrices B and A, with rank r much smaller than hidden dimension d. By training less than one percent of parameters, LoRA drastically reduces optimizer memory and folds back into base weights at inference with zero latency overhead. This establishes the foundation for examining how low-rank updates can be made even more expressive."
        ),
        # SLIDE 2
        (
            2,
            "Existing Research: Weight-Decomposed Low-Rank Adaptation (DoRA)",
            "60-65 seconds",
            "Moving to Slide 2, we examine Weight-Decomposed Low-Rank Adaptation, known as DoRA, published by Liu and colleagues in 2024.\n\n"
            "While LoRA is highly efficient, the authors discovered an inherent coupling limitation: in full fine-tuning, the magnitude of weight matrices and their directional orientation exhibit distinct, decoupled learning dynamics. In standard LoRA, delta-W couples magnitude and direction together proportionally, restricting optimization flexibility.\n\n"
            "DoRA resolves this through explicit weight decomposition, shown in our equation: the adapted weight matrix W equals a trainable magnitude vector m multiplied by the directional matrix V divided by its column-wise Euclidean norm. Here, V combines frozen base weights W_0 with the low-rank update delta-W. Normalizing V ensures unit directional columns.\n\n"
            "As illustrated on the right, the magnitude vector m independently tunes feature channel scaling, while matrices B and A learn directional rotations. This decoupling allows DoRA to mirror full fine-tuning dynamics closely. However, in standard DoRA, every adapted layer receives an identical, static rank budget, such as rank 3 across all projections. This rigid uniformity brings us directly to the research gap."
        ),
        # SLIDE 3
        (
            3,
            "Research Gap and Problem Statement",
            "55-60 seconds",
            "On Slide 3, we identify the central research gap that motivates our proposed work.\n\n"
            "Consider the three limitations on the slide. First, standard practice assigns an identical, uniform rank to every projection across all layers. Yet Transformer layers are fundamentally heterogeneous: lower layers capture localized syntax, while deeper layers process complex task semantics. Uniform allocation over-allocates capacity to low-demand layers while starving bottleneck layers.\n\n"
            "Second, consider existing dynamic pruning methods. While pruning removes low-utility components during training, it is strictly subtractive. Once a rank component is eliminated, that capacity is permanently discarded. It cannot be reinvested into other layers that remain capacity-starved.\n\n"
            "This leads directly to the unaddressed need: certain critical layers experience intense adaptation demand, but cannot acquire additional rank under existing static setups.\n\n"
            "This motivates our core research question shown below: How can a fixed low-rank parameter budget be allocated across Transformer layers according to their adaptation requirements, rather than relying only on uniform allocation or rank removal? Our objective is to propose a budget-constrained dynamic rank reallocation framework that transfers low-rank capacity from less useful layers to layers with greater adaptation demand."
        ),
        # SLIDE 4
        (
            4,
            "Proposed Framework: Dynamic Rank Reallocation",
            "50-55 seconds",
            "Turning to Slide 4, we introduce the core philosophy of our proposed framework: Dynamic Rank Reallocation.\n\n"
            "The central idea is to shift low-rank adaptation from a purely subtractive pruning paradigm into an autonomous, capacity-conserving reallocation mechanism. Instead of permanently discarding parameters that show low utility, our framework designates low-utility layers as eligible Donors. Simultaneously, layers demonstrating high adaptation demand are designated as eligible Recipients.\n\n"
            "Let us walk through the illustrative three-layer example on the right. Suppose the network operates under a total rank budget of 6, with Layer 1 at rank 3, Layer 2 at rank 2, and Layer 3 at rank 1. During training, if monitoring signals show Layer 1 has low adaptation utility while Layer 3 experiences intense gradient demand, the framework transfers one rank component from Layer 1 to Layer 3. The new allocation becomes ranks 2, 2, and 2. Crucially, the total active rank budget across the network remains exactly 6. The global parameter footprint is strictly conserved.\n\n"
            "This toy example illustrates the redistribution principle, which our proposed framework scales across all 72 adaptive projections of a Transformer encoder."
        ),
        # SLIDE 5
        (
            5,
            "Measuring Adaptation Requirements",
            "55-60 seconds",
            "Slide 5 details the mathematical signals our proposed framework uses to quantify layer adaptation requirements.\n\n"
            "To make principled reallocation decisions, we introduce a dual-signal measurement strategy combining retrospective component utility with prospective gradient demand.\n\n"
            "The first signal, on the left, is Component Importance, denoted D_l. For each layer l, D_l is computed as the average importance of its currently active low-rank components, dividing the sum of importance scores I_{l,j} by active rank r_l. Each score is derived from the relative Frobenius norm contribution of that low-rank component to the directional update. This provides a retrospective evaluation of how significantly existing components contributed to recent forward passes.\n\n"
            "However, retrospective importance cannot tell us if a layer needs more capacity. Therefore, our second signal on the right is Gradient Demand, denoted G_l. We compute the Frobenius norm of gradient matrices for adapter weights A_l and B_l, normalized by the square root of their parameter counts. This dimensional normalization prevents larger projection matrices from dominating purely due to matrix size. Gradient demand measures the prospective pull that the loss function is exerting on that layer.\n\n"
            "Combining both signals provides a balanced diagnostic: we identify layers under-utilizing their current rank alongside layers with urgent adaptation needs."
        ),
        # SLIDE 6
        (
            6,
            "Utility Estimation and Parameter-Aware Allocation",
            "55-60 seconds",
            "On Slide 6, we explain how these raw signals are synthesized into actionable reallocation decisions through temporal smoothing, utility combination, and parameter cost awareness.\n\n"
            "First, in Card 1, raw mini-batch gradients and instantaneous importance scores fluctuate due to batch sampling. To prevent erratic rank thrashing, we apply Exponential Moving Average smoothing with decay beta equal to 0.9. Both D_l and G_l are filtered over optimization steps t, creating stable historical trendlines.\n\n"
            "Second, in Card 2, we synthesize the smoothed signals into a unified Layer Utility score, U_l. We normalize both signals across all eligible layers using min-max scaling, and combine them using weighting parameter lambda equal to 0.5, giving balanced 50/50 weight to prospective gradient demand and retrospective importance.\n\n"
            "Finally, Card 3 introduces Parameter Cost Awareness. Adding one rank component to a linear layer costs d_in plus d_out plus 1 parameters. In RoBERTa, an attention projection costs 1,537 parameters per rank, whereas an intermediate MLP projection costs 3,841 parameters per rank—two and a half times more! Therefore, we define Cost-Adjusted Utility Q_l as raw utility U_l divided by parameter cost C_l. This prioritizes rank transfers to layers that deliver the highest adaptation utility per parameter invested."
        ),
        # SLIDE 7
        (
            7,
            "Dynamic Rank Reallocation Algorithm",
            "60-65 seconds",
            "Slide 7 brings these mathematical concepts together into our complete Dynamic Rank Reallocation Algorithm.\n\n"
            "Let us walk through the execution flowchart on the left. During fine-tuning, every 100 optimizer steps, a reallocation trigger is reached. The algorithm enters Phase 1, computing component importance D_l and gradient demand G_l across all layers, and updating their exponential moving averages.\n\n"
            "In Phase 2, the framework computes normalized layer utility U_l and cost-adjusted utility Q_l. In Phase 3, candidates are selected using the eligibility rules on the right: the Donor candidate is the layer with lowest utility U_l that has active rank strictly above r_min and satisfies the 500-step cooldown. The Recipient candidate is the layer with highest cost-adjusted utility Q_l that has spare capacity below r_max and satisfies cooldown.\n\n"
            "In Phase 4, the algorithm verifies stability guards: the utility margin between recipient and donor must exceed 0.1. If satisfied, an atomic transfer occurs: the least important component in the donor is deactivated, and a newly allocated slot in the recipient is activated.\n\n"
            "Crucially, donor weight tensors are never copied into the recipient layer because different layers have incompatible latent representations. Instead, the recipient slot is freshly initialized and trained, with its optimizer momentum buffers cleanly reset. Finally, the algorithm verifies that the global budget invariant is preserved, and training resumes seamlessly."
        ),
        # SLIDE 8
        (
            8,
            "Rank Budget, Constraints and Stability",
            "50-55 seconds",
            "Moving to Slide 8, we examine the formal constraints, hyperparameter bounds, and stability safeguards that ensure controlled optimization.\n\n"
            "On the left, we highlight the central mathematical invariant: the sum of active ranks across all L layers must equal the target budget B_t at every step. Reallocation operates orthogonally to scheduled global pruning: if a baseline pruning schedule decays the global budget from 216 to 144, B_t follows that scheduled curve. However, during any reallocation event, exactly one rank is subtracted from a donor and exactly one rank is added to a recipient, guaranteeing that the net change in total rank is strictly zero.\n\n"
            "In the table on the right, we outline our proposed hyperparameter bounds. Each layer is initialized with rank 3 across 72 projections, totaling 216 initial components. We enforce a strict minimum rank bound of 1, preventing catastrophic layer death. We set a maximum physical capacity of 4, providing headroom to receive capacity. Reallocation checks occur every 100 optimizer steps, with a cooldown of 500 steps per layer.\n\n"
            "These stability safeguards are vital: the transfer margin threshold of 0.1 prevents architectural oscillation over negligible utility differences, the cooldown period eliminates ping-pong rank bouncing, and limiting reallocation to at most one transfer per event ensures smooth optimization."
        ),
        # SLIDE 9
        (
            9,
            "Proposed Experimental Evaluation",
            "55-60 seconds",
            "On Slide 9, we outline the planned experimental evaluation designed to empirically test our proposed framework against matched baselines.\n\n"
            "As detailed on the left, our planned benchmark protocol utilizes roberta-base, adapting all 72 linear projections across its 12 encoder layers on the GLUE Stanford Sentiment Treebank dataset. Training is standardized to 30 epochs with AdamW, learning rate 8e-4, and batch size 64. The global rank budget begins at 216 and targets a final budget of 144, representing an average rank of 2.0 per layer.\n\n"
            "To isolate the true effect of dynamic reallocation, we design four strictly matched comparison groups shown on the right, all operating under the identical final rank budget of 144. Group 1 is Uniform Static Rank, fixing rank 2 across all layers. Group 2 is the existing Dynamic Pruning baseline, which prunes from 216 down to 144 without reinvestment. Group 3 is our Proposed Dynamic Reallocation framework, combining dual-signal utility with parameter cost awareness. Finally, Group 4 is Random Reallocation, an ablation control where transfers occur randomly between eligible layers.\n\n"
            "We plan to measure accuracy, macro-F1, final rank distributions, transfer frequencies, parameter storage in megabytes, training overhead, and inference latency.\n\n"
            "I want to explicitly reiterate the scientific integrity statement in red: this is a proposed experimental protocol. The reallocation extension is currently a theoretical design that has not yet been experimentally executed, and no experimental superiority is claimed."
        ),
        # SLIDE 10
        (
            10,
            "Expected Contributions, Limitations and Conclusion",
            "50-55 seconds",
            "To conclude our presentation on Slide 10, let us summarize our proposed research contributions, working hypotheses, and open limitations.\n\n"
            "As outlined in the first column, our primary theoretical contribution is proposing a principled, budget-preserving rank reallocation framework. We introduce a dual-signal adaptation measurement combining retrospective component utility with prospective gradient demand, normalize this by physical parameter cost, and enforce strict budget conservation through stability safeguards.\n\n"
            "In the center column, our working hypotheses propose that dynamic redistribution will allow the network to autonomously discover specialized layer-wise importance patterns—such as allocating greater rank capacity to deeper attention layers—thereby achieving superior parameter-efficiency trade-offs compared to uniform or purely subtractive baselines.\n\n"
            "Equally important, in the right column, we transparently state the scientific limitations of our proposal. This framework has not yet been implemented or experimentally validated; its empirical advantages, computational overhead from gradient tracking, and sensitivity to hyperparameters such as the transfer margin and cooldown remain to be verified.\n\n"
            "In conclusion, returning to our central research question: parameter-efficient fine-tuning need not be constrained by rigid uniform allocations or one-way parameter pruning. By formulating low-rank adaptation as a dynamic, budget-conserving reallocation framework, this research establishes a principled pathway toward optimal, autonomous parameter distribution in foundation models. Thank you for your time, and I welcome your questions and discussion."
        )
    ]

    for item in scripts:
        slide_num, title, est_time, text_content = item
        
        h = doc.add_heading(f"Slide {slide_num} - {title}", level=1)
        h.paragraph_format.space_before = DocxPt(16)
        h.paragraph_format.space_after = DocxPt(4)
        
        p_time = doc.add_paragraph()
        p_time.paragraph_format.space_after = DocxPt(10)
        r_time_lbl = p_time.add_run("Estimated Speaking Duration: ")
        r_time_lbl.font.bold = True
        r_time_lbl.font.size = DocxPt(10)
        r_time_lbl.font.color.rgb = DocxRGBColor(2, 132, 199)
        r_time_val = p_time.add_run(f"{est_time}  |  Slide {slide_num} of 10")
        r_time_val.font.size = DocxPt(10)
        r_time_val.font.color.rgb = DocxRGBColor(71, 85, 105)

        p_wts = doc.add_paragraph()
        p_wts.paragraph_format.space_after = DocxPt(6)
        r_wts = p_wts.add_run("What to say:")
        r_wts.font.bold = True
        r_wts.font.size = DocxPt(11)
        r_wts.font.color.rgb = DocxRGBColor(15, 23, 42)

        paragraphs = text_content.split("\n\n")
        for p_text in paragraphs:
            p_body = doc.add_paragraph()
            p_body.paragraph_format.space_after = DocxPt(8)
            p_body.paragraph_format.line_spacing = 1.15
            r_body = p_body.add_run(p_text.strip())
            r_body.font.name = "Calibri"
            r_body.font.size = DocxPt(11)
            r_body.font.color.rgb = DocxRGBColor(30, 41, 59)

        if slide_num < 10:
            p_div = doc.add_paragraph()
            p_div.paragraph_format.space_before = DocxPt(10)
            p_div.paragraph_format.space_after = DocxPt(14)
            r_div = p_div.add_run("—" * 55)
            r_div.font.color.rgb = DocxRGBColor(203, 213, 225)

    docx_path = out_dir / "AI_DoRA_Presentation_Script.docx"
    doc.save(docx_path)
    print(f"Script saved to {docx_path} (Sections: {len(scripts)})")


if __name__ == "__main__":
    create_presentation()
    create_script()
