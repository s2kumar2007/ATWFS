"""Generate high-quality ATWFS PowerPoint presentation matching original PDF template and structure."""
import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# Color Palette Constants
COLOR_TITLE = RGBColor(0x1A, 0x1A, 0x1A)       # Dark typewriter charcoal
COLOR_SUBTITLE = RGBColor(0x2E, 0x66, 0x48)    # Sage / forest green
COLOR_BODY = RGBColor(0x26, 0x26, 0x26)        # Neutral readable dark grey
COLOR_LEAD = RGBColor(0x11, 0x11, 0x11)        # Bold lead-in black
COLOR_BORDER_GREEN = RGBColor(0x2E, 0x6B, 0x52)
COLOR_FILL_GREEN = RGBColor(0xF4, 0xF9, 0xF5)
COLOR_BORDER_GOLD = RGBColor(0xB8, 0x8F, 0x28)
COLOR_FILL_GOLD = RGBColor(0xFA, 0xF6, 0xEB)
COLOR_BORDER_DARK = RGBColor(0x1B, 0x4D, 0x3A)
COLOR_FILL_DARK = RGBColor(0xEE, 0xF5, 0xF1)
COLOR_ARROW = RGBColor(0x33, 0x33, 0x33)

FONT_TITLE = "Courier New"
FONT_BODY = "Segoe UI"
FONT_CODE = "Consolas"

def create_presentation(assets_dir: str, output_path: str):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_base_slide(page_num: int):
        slide = prs.slides.add_slide(blank_layout)
        bg_file = os.path.join(assets_dir, f"page_{page_num}_img_1.jpeg")
        if os.path.exists(bg_file):
            slide.shapes.add_picture(bg_file, 0, 0, width=prs.slide_width, height=prs.slide_height)
        # Add slide number on content pages (2..12)
        if 2 <= page_num <= 12:
            num_box = slide.shapes.add_textbox(Inches(12.0), Inches(6.85), Inches(0.8), Inches(0.4))
            tf = num_box.text_frame
            tf.word_wrap = False
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT
            p.text = str(page_num)
            p.font.name = FONT_TITLE
            p.font.size = Pt(12)
            p.font.color.rgb = COLOR_TITLE
        return slide

    def add_title(slide, text: str):
        box = slide.shapes.add_textbox(Inches(1.35), Inches(0.55), Inches(11.0), Inches(0.9))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = text
        p.font.name = FONT_TITLE
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = COLOR_TITLE

    def add_box(slide, left, top, width, height, title, subtitle=None, style="green"):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.shadow.inherit = False
        border_c = COLOR_BORDER_GOLD if style == "gold" else (COLOR_BORDER_DARK if style == "dark" else COLOR_BORDER_GREEN)
        fill_c = COLOR_FILL_GOLD if style == "gold" else (COLOR_FILL_DARK if style == "dark" else COLOR_FILL_GREEN)
        shape.line.color.rgb = border_c
        shape.line.width = Pt(1.5)
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_c
        tf = shape.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = title
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_TITLE
        if subtitle:
            p2 = tf.add_paragraph()
            p2.alignment = PP_ALIGN.CENTER
            p2.text = subtitle
            p2.font.name = FONT_BODY
            p2.font.size = Pt(10.5)
            p2.font.color.rgb = COLOR_BODY
        return shape

    def add_down_arrow(slide, center_x, top_y, length=Inches(0.28)):
        arrow = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, center_x - Inches(0.08), top_y, Inches(0.16), length)
        arrow.shadow.inherit = False
        arrow.line.color.rgb = COLOR_ARROW
        arrow.line.width = Pt(1.0)
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = COLOR_ARROW

    def add_right_arrow(slide, left_x, center_y, length=Inches(0.4)):
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left_x, center_y - Inches(0.08), length, Inches(0.16))
        arrow.shadow.inherit = False
        arrow.line.color.rgb = COLOR_ARROW
        arrow.line.width = Pt(1.0)
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = COLOR_ARROW

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    s1 = add_base_slide(1)
    t_box = s1.shapes.add_textbox(Inches(1.35), Inches(1.3), Inches(8.5), Inches(3.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "Vision-Based Road\nDrivable Region\nEstimation for Lane-Less\nRoads"
    p.font.name = FONT_TITLE
    p.font.size = Pt(31)
    p.font.bold = True
    p.font.color.rgb = COLOR_TITLE

    p_sub = tf1.add_paragraph()
    p_sub.text = "with Adaptive Trust-Weighted Fusion Segmentation (ATWFS)"
    p_sub.font.name = FONT_BODY
    p_sub.font.size = Pt(16)
    p_sub.font.italic = True
    p_sub.font.color.rgb = COLOR_SUBTITLE
    p_sub.space_before = Pt(14)

    a_box = s1.shapes.add_textbox(Inches(1.35), Inches(4.7), Inches(8.5), Inches(2.2))
    tfa = a_box.text_frame
    tfa.word_wrap = True
    p_course = tfa.paragraphs[0]
    p_course.text = "BACSE324 · Embedded Systems Architecture"
    p_course.font.name = FONT_BODY
    p_course.font.size = Pt(14)
    p_course.font.bold = True
    p_course.font.color.rgb = COLOR_SUBTITLE

    authors = [
        "Sunil Kumar  –  25BCE0388",
        "Varshith R S –  25BCE0351",
        "Suyash Gupta –  25BCE0371",
    ]
    for auth in authors:
        pa = tfa.add_paragraph()
        pa.text = auth
        pa.font.name = FONT_BODY
        pa.font.size = Pt(13.5)
        pa.font.color.rgb = COLOR_BODY
        pa.space_before = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 2: Introduction
    # -------------------------------------------------------------
    s2 = add_base_slide(2)
    add_title(s2, "Introduction")
    b2 = s2.shapes.add_textbox(Inches(1.35), Inches(1.7), Inches(6.8), Inches(5.0))
    tf2 = b2.text_frame
    tf2.word_wrap = True

    intro_points = [
        ("Unstructured Road Topologies:", " Lane-less rural and semi-urban roads lack demarcated painted lines, exhibiting irregular asphalt boundaries, earthen shoulders, and dirt transitions that cause conventional lane-tracking systems to fail."),
        ("Inference Disconnect in Deep Models:", " Modern deep segmentation networks typically terminate at passive 2D probability masks or isolated steering angles without closed-loop kinematic coupling to vehicle dynamics."),
        ("OOD Fallibility & Calibration Absence:", " Off-the-shelf vision systems lack dynamic camera pitch/roll self-calibration and provide no fail-safe fallback mechanism when encountering out-of-distribution (OOD) visual anomalies."),
        ("Closed-Loop Monocular Objective:", " Develop a camera-only, real-time edge architecture that concurrently delivers metric drivable space estimation, dynamic obstacle avoidance, and coupled speed/steering control."),
    ]
    for i, (lead, body) in enumerate(intro_points):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = "• " + lead
        r1.font.name = FONT_BODY
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_LEAD
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_BODY
        r2.font.size = Pt(12.5)
        r2.font.color.rgb = COLOR_BODY

    # Flowchart on right
    fc_x = Inches(8.8)
    fc_w = Inches(3.8)
    fc_h = Inches(0.95)
    gap = Inches(0.38)
    y_start = Inches(1.7)

    add_box(s2, fc_x, y_start, fc_w, fc_h, "Camera Frame", "Monocular RGB dashcam stream", style="green")
    add_down_arrow(s2, fc_x + fc_w / 2, y_start + fc_h, gap)
    y_start += fc_h + gap
    add_box(s2, fc_x, y_start, fc_w, fc_h, "ATWFS Pipeline", "Hybrid DL segmentation + Evidential trust", style="gold")
    add_down_arrow(s2, fc_x + fc_w / 2, y_start + fc_h, gap)
    y_start += fc_h + gap
    add_box(s2, fc_x, y_start, fc_w, fc_h, "Drivable Mask", "Metric free-space corridor & obstacle gating", style="green")
    add_down_arrow(s2, fc_x + fc_w / 2, y_start + fc_h, gap)
    y_start += fc_h + gap
    add_box(s2, fc_x, y_start, fc_w, fc_h, "Speed + Steering", "Kinematically clamped control packet", style="dark")

    # -------------------------------------------------------------
    # SLIDE 3: Literature Review
    # -------------------------------------------------------------
    s3 = add_base_slide(3)
    add_title(s3, "Literature Review")
    b3 = s3.shapes.add_textbox(Inches(1.35), Inches(1.65), Inches(6.8), Inches(5.2))
    tf3 = b3.text_frame
    tf3.word_wrap = True

    p0 = tf3.paragraphs[0]
    p0.text = "Existing Methods & Paradigms:"
    p0.font.name = FONT_BODY
    p0.font.bold = True
    p0.font.size = Pt(14)
    p0.font.color.rgb = COLOR_TITLE
    p0.space_after = Pt(8)

    lit_points = [
        ("Classical Computer Vision (Canny/Hough):", " Highly compute-efficient for embedded systems, but fundamentally relies on high-contrast painted lane lines; collapses on broken tar, dirt roads, or mud."),
        ("Deep Semantic Segmentation (U-Net/SegNet):", " Robust feature representation on complex road surfaces, but computationally heavy and typically outputs raw pixel masks without driving commands."),
        ("Direct Mask-to-Steering (Rasib et al., 2021):", " Couples drivable masks to steering commands via geometric path extraction, but lacks uncertainty awareness and provides zero fallback when the network is blind."),
        ("Surface Normal Multi-Modal (SNE-RoadSegV2):", " Introduces fallibility and surface-normal awareness, but strictly depends on costly LiDAR point clouds or calibrated stereo camera pairs."),
    ]
    for lead, body in lit_points:
        p = tf3.add_paragraph()
        p.space_after = Pt(10)
        r1 = p.add_run()
        r1.text = "• " + lead
        r1.font.name = FONT_BODY
        r1.font.bold = True
        r1.font.size = Pt(12.5)
        r1.font.color.rgb = COLOR_LEAD
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_BODY
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_BODY

    # Flowchart on right
    y_start = Inches(1.7)
    add_box(s3, fc_x, y_start, fc_w, fc_h, "Classical CV", "Fast edge/line detection; fragile to missing markings", style="green")
    add_down_arrow(s3, fc_x + fc_w / 2, y_start + fc_h, gap)
    y_start += fc_h + gap
    add_box(s3, fc_x, y_start, fc_w, fc_h, "Deep Segmentation", "High semantic accuracy; passive, uncalibrated", style="green")
    add_down_arrow(s3, fc_x + fc_w / 2, y_start + fc_h, gap)
    y_start += fc_h + gap
    add_box(s3, fc_x, y_start, fc_w, fc_h, "Mask + Steering Only", "Geometric control derivation; lacks safety fallback", style="gold")
    add_down_arrow(s3, fc_x + fc_w / 2, y_start + fc_h, gap)
    y_start += fc_h + gap
    add_box(s3, fc_x, y_start, fc_w, fc_h, "ATWFS (Ours)", "Monocular, self-calibrating, uncertainty-aware trust fusion", style="dark")

    # -------------------------------------------------------------
    # SLIDE 4: Research Gap
    # -------------------------------------------------------------
    s4 = add_base_slide(4)
    add_title(s4, "Research Gap")
    b4 = s4.shapes.add_textbox(Inches(1.35), Inches(1.65), Inches(5.8), Inches(5.2))
    tf4 = b4.text_frame
    tf4.word_wrap = True

    gap_points = [
        ("Dynamic Pitch/Roll Vulnerability:", " Vehicle motion over uneven terrain continuously alters camera orientation, distorting metric distance mapping when using fixed perspective calibration."),
        ("Uncalibrated Model Confidence:", " Standard neural networks suffer from overconfidence, producing erroneous predictions on out-of-distribution road domains without warning."),
        ("Absence of Pre-Actuation Safety Verification:", " Steering and throttle commands are frequently driven directly by unverified network outputs without geometric or temporal consistency safety checks."),
        ("Prohibitive Multi-Sensor Hardware Costs:", " High reliance on multi-beam LiDAR and stereo setups prevents practical adoption on low-cost edge vehicles."),
        ("Extreme Environmental Fragility:", " Off-the-shelf models trained on Western highways degrade severely on chaotic Indian roads featuring stray cattle, dust, glare, and irregular road edges."),
    ]
    for i, (lead, body) in enumerate(gap_points):
        p = tf4.paragraphs[0] if i == 0 else tf4.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + lead
        r1.font.name = FONT_BODY
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = COLOR_LEAD
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_BODY
        r2.font.size = Pt(11.5)
        r2.font.color.rgb = COLOR_BODY

    # Comparison Pairs on right
    pair_y = Inches(1.7)
    pair_h = Inches(1.0)
    pair_gap = Inches(0.3)
    p_w1 = Inches(2.2)
    p_w2 = Inches(3.0)
    arr_len = Inches(0.4)
    x_left_box = Inches(7.3)
    x_arrow = x_left_box + p_w1 + Inches(0.1)
    x_right_box = x_arrow + arr_len + Inches(0.1)

    pairs = [
        ("No calibration", "Fixed camera height assumption", "IMU-based IPM", "Self-calibrating pitch/roll homography"),
        ("No fallback", "Blind neural trust on edge cases", "Trust-weighted blend", "Continuous Dirichlet evidential fusion"),
        ("No safety check", "Unverified actuator commands", "Geometric & temporal checks", "Perspective validity & optical flow gating"),
        ("Costly sensors", "Expensive LiDAR / stereo rigs", "Monocular camera", "Standard dashcam on low-cost edge CPU"),
    ]
    for (l_tit, l_sub, r_tit, r_sub) in pairs:
        add_box(s4, x_left_box, pair_y, p_w1, pair_h, l_tit, l_sub, style="gold")
        add_right_arrow(s4, x_arrow, pair_y + pair_h / 2, length=arr_len)
        add_box(s4, x_right_box, pair_y, p_w2, pair_h, r_tit, r_sub, style="green")
        pair_y += pair_h + pair_gap

    # -------------------------------------------------------------
    # SLIDE 5: Problem Formulation
    # -------------------------------------------------------------
    s5 = add_base_slide(5)
    add_title(s5, "Problem Formulation")
    b5 = s5.shapes.add_textbox(Inches(1.35), Inches(1.7), Inches(10.8), Inches(5.2))
    tf5 = b5.text_frame
    tf5.word_wrap = True

    prob_points = [
        ("1. Sequential Monocular Input Stream:", " Continuous video stream I_t in R^(H x W x 3) captured from an uncalibrated forward dashcam operating under uncontrolled lighting, shadow transitions, and camera vibrations."),
        ("2. Multi-Task Output Tuple:", " Concurrent prediction of binary drivable free-space mask M_t in {0, 1}^(H x W), closed-form evidential epistemic uncertainty U_t in (0, 1], and discrete surface classification C_terrain in {dry tar, wet/mud, gravel, waterlogged}."),
        ("3. Dynamic In-Path Obstacle Subtraction:", " Active identification and spatial subtraction of critical dynamic obstacles (pedestrians, vehicles, stray cattle) from the drivable corridor using safety-dilated bounding boxes."),
        ("4. Coupled Speed & Steering Kinematics:", " Simultaneous computation of steering angle theta_t in [-30°, +30°] via pure pursuit trajectory tracking and target velocity v_t in [0.0, 1.5] m/s constrained by road width, curvature, and obstacle proximity."),
        ("5. Trust-Mediated Dynamic Fallback:", " Continuous, smooth transition to a deterministic classical color-space (HSV/Lab) and morphological flood-fill estimator whenever neural trust drops below safe operational margins."),
    ]
    for i, (lead, body) in enumerate(prob_points):
        p = tf5.paragraphs[0] if i == 0 else tf5.add_paragraph()
        p.space_after = Pt(14)
        r1 = p.add_run()
        r1.text = "• " + lead
        r1.font.name = FONT_BODY
        r1.font.bold = True
        r1.font.size = Pt(13.5)
        r1.font.color.rgb = COLOR_LEAD
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_BODY
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_BODY

    # -------------------------------------------------------------
    # SLIDE 6: Proposed System
    # -------------------------------------------------------------
    s6 = add_base_slide(6)
    add_title(s6, "Proposed System")
    b6 = s6.shapes.add_textbox(Inches(1.35), Inches(1.65), Inches(6.8), Inches(5.2))
    tf6 = b6.text_frame
    tf6.word_wrap = True

    prop_points = [
        ("Self-Calibrating IPM from IMU Pitch/Roll:", " Dynamically updates homography matrices using vehicle IMU telemetry and vanishing-point geometry to maintain an accurate metric Bird's-Eye-View (BEV) road grid (10 m x 25 m)."),
        ("YOLOv8n Dynamic Obstacle Subtraction:", " Lightweight single-pass detector detects pedestrians, oncoming traffic, and stray animals; dilates their contact footprints to subtract them from the drivable corridor."),
        ("Multi-Factor Trust Arbitration:", " Combines deep learning segmentation with classical HSV/Lab color flood-fill based on evidential uncertainty, mask shape geometry, and temporal frame-to-frame continuity."),
        ("Coupled Speed and Steering Decision:", " Jointly derives forward speed and steering angle, enforcing physical rate limits (5°/frame) and speed-dependent minimum turn radii (R_min = v^2 / a_lat)."),
        ("Deterministic Telemetry Packet:", " Generates a standardized 4-field control packet: [angle, speed, trust, obstacle_flag] transmitted over serial to low-level vehicle actuators."),
    ]
    for i, (lead, body) in enumerate(prop_points):
        p = tf6.paragraphs[0] if i == 0 else tf6.add_paragraph()
        p.space_after = Pt(10)
        r1 = p.add_run()
        r1.text = "• " + lead
        r1.font.name = FONT_BODY
        r1.font.bold = True
        r1.font.size = Pt(12.5)
        r1.font.color.rgb = COLOR_LEAD
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_BODY
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_BODY

    # Flowchart on right (5 boxes)
    fc_h5 = Inches(0.8)
    gap5 = Inches(0.24)
    y_start = Inches(1.65)
    add_box(s6, fc_x, y_start, fc_w, fc_h5, "Camera + IMU", "Synchronized monocular video & 6-DOF inertial data", style="green")
    add_down_arrow(s6, fc_x + fc_w / 2, y_start + fc_h5, gap5)
    y_start += fc_h5 + gap5
    add_box(s6, fc_x, y_start, fc_w, fc_h5, "Self-Calibrating IPM", "Real-time pitch/roll homography projection", style="green")
    add_down_arrow(s6, fc_x + fc_w / 2, y_start + fc_h5, gap5)
    y_start += fc_h5 + gap5
    add_box(s6, fc_x, y_start, fc_w, fc_h5, "Geometric + Temporal Checks", "Perspective narrowing & flow consistency", style="green")
    add_down_arrow(s6, fc_x + fc_w / 2, y_start + fc_h5, gap5)
    y_start += fc_h5 + gap5
    add_box(s6, fc_x, y_start, fc_w, fc_h5, "ATWFS Trust Fusion", "Multi-signal arbitration: DL vs. Classical", style="gold")
    add_down_arrow(s6, fc_x + fc_w / 2, y_start + fc_h5, gap5)
    y_start += fc_h5 + gap5
    add_box(s6, fc_x, y_start, fc_w, fc_h5, "Speed + Steering Packet", "Safety-clamped kinematics to vehicle actuators", style="dark")

    # -------------------------------------------------------------
    # SLIDE 7: Model Architecture
    # -------------------------------------------------------------
    s7 = add_base_slide(7)
    add_title(s7, "Model Architecture")
    b7 = s7.shapes.add_textbox(Inches(1.35), Inches(1.65), Inches(6.8), Inches(5.2))
    tf7 = b7.text_frame
    tf7.word_wrap = True

    arch_points = [
        ("MobileNetV3-Large Encoder:", " Employs depthwise separable convolutions, inverted bottleneck blocks, and squeeze-and-excitation (SE) modules for efficient visual feature extraction on low-power edge CPUs."),
        ("GroupNorm Stability Throughout:", " Replaces standard BatchNorm2d with GroupNorm (8 groups per layer) throughout the network to eliminate batch-size dependency and ensure 100% deterministic inference across edge batch sizes."),
        ("Multi-Task U-Net Decoder:", " Decodes multiscale stride-[32, 16, 8, 4] skip features to concurrently output binary drivable logits and Dirichlet evidential concentration parameters alpha = softplus(z) + 1."),
        ("Stride-8 Auxiliary Terrain Head:", " Branches from intermediate encoder representations into a lightweight classification MLP predicting 4 surface types (dry tar, wet/mud, gravel, waterlogged) to inform traction control and speed derating."),
    ]
    for i, (lead, body) in enumerate(arch_points):
        p = tf7.paragraphs[0] if i == 0 else tf7.add_paragraph()
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = "• " + lead
        r1.font.name = FONT_BODY
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_LEAD
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_BODY
        r2.font.size = Pt(12.5)
        r2.font.color.rgb = COLOR_BODY

    # Architecture Diagram on right
    d_x = Inches(8.8)
    d_w = Inches(3.8)
    add_box(s7, d_x + Inches(0.4), Inches(1.65), Inches(3.0), Inches(0.85), "Input Image", "288 x 512 x 3 RGB Frame", style="green")
    add_down_arrow(s7, d_x + Inches(1.9), Inches(2.5), Inches(0.25))

    add_box(s7, d_x, Inches(2.75), d_w, Inches(1.0), "MobileNetV3 Encoder", "GroupNorm + Inverted Residuals", style="gold")

    # Split branches
    # Left branch: U-Net decoder
    b_w = Inches(1.8)
    add_box(s7, d_x - Inches(0.1), Inches(4.35), b_w, Inches(1.0), "U-Net Decoder", "Skip Connections", style="green")
    add_down_arrow(s7, d_x - Inches(0.1) + b_w / 2, Inches(5.35), Inches(0.25))
    add_box(s7, d_x - Inches(0.1), Inches(5.6), b_w, Inches(1.0), "Mask + Uncertainty", "Logits & Dirichlet alpha", style="gold")

    # Right branch: Terrain head
    add_box(s7, d_x + Inches(2.1), Inches(4.35), b_w, Inches(1.0), "Terrain Head", "Stride-8 Features", style="green")
    add_down_arrow(s7, d_x + Inches(2.1) + b_w / 2, Inches(5.35), Inches(0.25))
    add_box(s7, d_x + Inches(2.1), Inches(5.6), b_w, Inches(1.0), "Terrain Type", "4 Surface Classes", style="gold")

    # Connector lines from encoder to branches
    # Add downward left and right arrows
    arr_l = s7.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, d_x + Inches(0.8), Inches(3.75), Inches(0.14), Inches(0.55))
    arr_l.line.color.rgb = COLOR_ARROW
    arr_l.fill.solid()
    arr_l.fill.fore_color.rgb = COLOR_ARROW
    arr_r = s7.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, d_x + Inches(2.8), Inches(3.75), Inches(0.14), Inches(0.55))
    arr_r.line.color.rgb = COLOR_ARROW
    arr_r.fill.solid()
    arr_r.fill.fore_color.rgb = COLOR_ARROW

    # -------------------------------------------------------------
    # SLIDE 8: ATWFS: Trust Fusion
    # -------------------------------------------------------------
    s8 = add_base_slide(8)
    add_title(s8, "ATWFS: Trust Fusion")
    b8 = s8.shapes.add_textbox(Inches(1.35), Inches(1.65), Inches(6.8), Inches(5.2))
    tf8 = b8.text_frame
    tf8.word_wrap = True

    trust_points = [
        ("Multi-Signal Reliability Metric:", " Evaluates network reliability using three independent signals: evidential uncertainty (U), geometric perspective validity (G), and temporal consistency (T)."),
        ("Adaptive MLP Weight Generation:", " A lightweight neural fusion layer maps (U, G, T, P_terrain) to an optimal scalar blend factor alpha in [0, 1]."),
        ("Terrain-Aware Risk Derating:", " Slippery or unpaved terrain categories penalize the trust score, demanding higher certainty before trusting the deep model."),
    ]
    for i, (lead, body) in enumerate(trust_points):
        p = tf8.paragraphs[0] if i == 0 else tf8.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + lead
        r1.font.name = FONT_BODY
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = COLOR_LEAD
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_BODY
        r2.font.size = Pt(11.5)
        r2.font.color.rgb = COLOR_BODY

    # Mathematical Formula Box
    f_box = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.35), Inches(4.0), Inches(6.8), Inches(1.6))
    f_box.line.color.rgb = COLOR_BORDER_GREEN
    f_box.line.width = Pt(1.5)
    f_box.fill.solid()
    f_box.fill.fore_color.rgb = COLOR_FILL_GREEN
    tff = f_box.text_frame
    tff.vertical_anchor = MSO_ANCHOR.MIDDLE
    tff.word_wrap = True

    pf1 = tff.paragraphs[0]
    pf1.text = "Z = 12.0 × [0.50(1 - U) + 0.25G + 0.25T - 0.75 - P_terrain]"
    pf1.font.name = FONT_CODE
    pf1.font.size = Pt(11.5)
    pf1.font.bold = True
    pf1.font.color.rgb = COLOR_SUBTITLE

    pf2 = tff.add_paragraph()
    pf2.text = "Trust (α) = (1 + e^-Z)^-1"
    pf2.font.name = FONT_CODE
    pf2.font.size = Pt(12)
    pf2.font.bold = True
    pf2.font.color.rgb = COLOR_SUBTITLE
    pf2.space_before = Pt(4)

    pf3 = tff.add_paragraph()
    pf3.text = "Final Mask = α · DL mask + (1 - α) · Classical Fallback"
    pf3.font.name = FONT_CODE
    pf3.font.size = Pt(11.5)
    pf3.font.bold = True
    pf3.font.color.rgb = COLOR_TITLE
    pf3.space_before = Pt(4)

    # Novelty box
    n_box = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.35), Inches(5.8), Inches(6.8), Inches(0.9))
    n_box.line.color.rgb = COLOR_BORDER_GOLD
    n_box.line.width = Pt(1.5)
    n_box.fill.solid()
    n_box.fill.fore_color.rgb = COLOR_FILL_GOLD
    tfn = n_box.text_frame
    tfn.vertical_anchor = MSO_ANCHOR.MIDDLE
    pn = tfn.paragraphs[0]
    pn.text = "Novelty: Learned, continuous blend — eliminates violent steering chatter and threshold oscillation."
    pn.font.name = FONT_BODY
    pn.font.size = Pt(11.5)
    pn.font.bold = True
    pn.font.color.rgb = COLOR_TITLE

    # Right Diagram for Trust Fusion
    in_box_w = Inches(1.9)
    in_box_h = Inches(0.7)
    y_in = Inches(1.7)
    in_labels = [
        ("U: Uncertainty", "Evidential Vacuity"),
        ("G: Geometry", "Perspective Trapezoid"),
        ("T: Temporal", "IoU Motion Continuity"),
        ("Terrain", "Friction Penalty"),
    ]
    for lbl, sub in in_labels:
        add_box(s8, Inches(8.5), y_in, in_box_w, in_box_h, lbl, sub, style="green")
        add_right_arrow(s8, Inches(10.5), y_in + in_box_h / 2, length=Inches(0.4))
        y_in += in_box_h + Inches(0.28)

    # Central Alpha Node
    add_box(s8, Inches(11.0), Inches(2.8), Inches(1.8), Inches(1.2), "Trust Weight (α)", "Sigmoid Fusion", style="gold")
    add_down_arrow(s8, Inches(11.9), Inches(4.0), Inches(0.35))
    add_box(s8, Inches(10.8), Inches(4.4), Inches(2.2), Inches(1.1), "Final Mask", "Fleshed Free-Space", style="dark")

    # -------------------------------------------------------------
    # SLIDE 9: Drivable Region with ATWFS
    # -------------------------------------------------------------
    s9 = add_base_slide(9)
    add_title(s9, "Drivable Region with ATWFS")
    # Insert high-resolution video frame
    demo_img_path = os.path.join(assets_dir, "page_9_img_3.jpeg")
    if os.path.exists(demo_img_path):
        s9.shapes.add_picture(demo_img_path, Inches(1.35), Inches(1.6), width=Inches(7.6), height=Inches(4.8))

    # Explanatory right panel
    p9_box = s9.shapes.add_textbox(Inches(9.2), Inches(1.6), Inches(3.8), Inches(4.8))
    tf9 = p9_box.text_frame
    tf9.word_wrap = True

    p9_title = tf9.paragraphs[0]
    p9_title.text = "In-Cab HUD & Actuation Breakdown:"
    p9_title.font.name = FONT_BODY
    p9_title.font.bold = True
    p9_title.font.size = Pt(13.5)
    p9_title.font.color.rgb = COLOR_TITLE
    p9_title.space_after = Pt(8)

    hud_notes = [
        ("Drivable Free-Space (Blue):", " Fused high-resolution surface mask filtered for sub-pixel boundary smoothness and zero edge flutter."),
        ("Hermite Trajectory (Yellow):", " S-curve spline anchored strictly to the ego bumper (0° launch) and converging to lane curvature at the horizon."),
        ("Seamless Arrowhead Alignment:", " Polyline terminates strictly into the arrowhead notch with continuous tangent vector alignment."),
        ("Real-Time Telemetry Box:", " Continuous display of active estimator (DL/Fallback), trust score (0.86), steering angle (-0.9°), road width (6.85 m), and 23.1 FPS throughput."),
    ]
    for lead, desc in hud_notes:
        p = tf9.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + lead
        r1.font.name = FONT_BODY
        r1.font.bold = True
        r1.font.size = Pt(11.5)
        r1.font.color.rgb = COLOR_LEAD
        r2 = p.add_run()
        r2.text = desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(11)
        r2.font.color.rgb = COLOR_BODY

    # -------------------------------------------------------------
    # SLIDE 10: Computational Efficiency
    # -------------------------------------------------------------
    s10 = add_base_slide(10)
    add_title(s10, "Computational Efficiency")

    # Table on Left
    tbl_shape = s10.shapes.add_table(7, 3, Inches(1.35), Inches(1.65), Inches(7.2), Inches(4.2))
    table = tbl_shape.table
    table.columns[0].width = Inches(3.0)
    table.columns[1].width = Inches(2.1)
    table.columns[2].width = Inches(2.1)

    table_data = [
        ("Metric", "Teacher", "Student"),
        ("Inference time (ms/frame)", "~110.8 ms/frame", "~31.8 – 45.1 ms/frame"),
        ("FLOPs (GFLOPs)", "8.78 GFLOPs", "3.23 GFLOPs"),
        ("Parameters (M)", "4.59 Million", "1.39 Million"),
        ("Throughput (FPS)", "9.0 FPS", "22.5 FPS"),
        ("End-to-end latency (ms)", "151.9 ms", "~44.4 ms"),
        ("Peak memory (MB)", "~1069.8 MB", "~893.4 MB"),
    ]
    for row_idx, row in enumerate(table_data):
        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.text = val
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cp = cell.text_frame.paragraphs[0]
            cp.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
            cp.font.name = FONT_BODY
            if row_idx == 0:
                cp.font.bold = True
                cp.font.size = Pt(12.5)
                cp.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0x22, 0x22, 0x22)
            else:
                cp.font.size = Pt(11.5)
                cp.font.color.rgb = COLOR_TITLE
                if col_idx == 2:
                    cp.font.bold = True
                    cp.font.color.rgb = COLOR_SUBTITLE

    b10_sub = s10.shapes.add_textbox(Inches(1.35), Inches(6.05), Inches(7.2), Inches(0.6))
    p_sub = b10_sub.text_frame.paragraphs[0]
    p_sub.text = "* Benchmarks measured on standard commodity x86 laptop CPU (no discrete GPU)."
    p_sub.font.name = FONT_BODY
    p_sub.font.italic = True
    p_sub.font.size = Pt(10.5)
    p_sub.font.color.rgb = COLOR_BODY

    # Benchmark Setup box on right
    b10_box = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.9), Inches(1.65), Inches(3.9), Inches(4.2))
    b10_box.line.color.rgb = COLOR_BORDER_GOLD
    b10_box.line.width = Pt(1.5)
    b10_box.fill.solid()
    b10_box.fill.fore_color.rgb = COLOR_FILL_GOLD
    tf10 = b10_box.text_frame
    tf10.word_wrap = True

    p10_h = tf10.paragraphs[0]
    p10_h.alignment = PP_ALIGN.CENTER
    p10_h.text = "Benchmark Setup & Analysis"
    p10_h.font.name = FONT_BODY
    p10_h.font.bold = True
    p10_h.font.size = Pt(13.5)
    p10_h.font.color.rgb = COLOR_TITLE
    p10_h.space_after = Pt(10)

    setup_points = [
        ("Teacher Model:", " MobileNetV3-Large MultiTaskUNet backbone with GroupNorm."),
        ("Student Model:", " MobileNetV3-Small MultiTaskUNet trained via knowledge distillation."),
        ("Input Resolution:", " 288 × 512 pixels matching the native optical aspect ratio."),
        ("Hardware Envelope:", " Standard low-to-medium tier laptop CPU (Intel Core i5)."),
        ("2.5x Speedup:", " Student achieves 22.5 FPS real-time throughput on edge hardware."),
        ("3.3x Compression:", " Memory footprint reduced from 52 MB to 16 MB checkpoint size."),
    ]
    for lead, desc in setup_points:
        p = tf10.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + lead
        r1.font.name = FONT_BODY
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = COLOR_LEAD
        r2 = p.add_run()
        r2.text = desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = COLOR_BODY

    # -------------------------------------------------------------
    # SLIDE 11: Dataset and Training
    # -------------------------------------------------------------
    s11 = add_base_slide(11)
    add_title(s11, "Dataset and Training")
    b11 = s11.shapes.add_textbox(Inches(1.35), Inches(1.65), Inches(6.8), Inches(5.2))
    tf11 = b11.text_frame
    tf11.word_wrap = True

    data_points = [
        ("Benchmark Corpus (IDD 20k Part I):", " Utilizes 10,003 unconstrained Indian road scenes capturing unstructured road edges, ambiguous margins, and heterogeneous traffic patterns."),
        ("Standardized Partitioning:", " Formally split into 6,993 training, 981 validation, and 2,029 testing images for reproducible benchmarking."),
        ("Semantic Class Collapse:", " Aggregates fine-grained 41 semantic classes into binary free-space (level3Ids {0, 1} -> drivable, all valid obstacles -> non-drivable, 255 ignored)."),
        ("Custom Edge-Case Collection:", " Augmented with custom dashcam footage annotated in CVAT covering low-light night driving, muddy unpaved roads, and stray cattle."),
        ("Independent Test Verification:", " Evaluated on an isolated held-out test split ensuring strict zero data leakage."),
    ]
    for i, (lead, body) in enumerate(data_points):
        p = tf11.paragraphs[0] if i == 0 else tf11.add_paragraph()
        p.space_after = Pt(10)
        r1 = p.add_run()
        r1.text = "• " + lead
        r1.font.name = FONT_BODY
        r1.font.bold = True
        r1.font.size = Pt(12.5)
        r1.font.color.rgb = COLOR_LEAD
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_BODY
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_BODY

    # Training Pipeline Flowchart on right
    fc_w6 = Inches(2.2)
    fc_h6 = Inches(0.85)
    gap_y = Inches(0.4)
    x_c1 = Inches(8.5)
    x_c2 = Inches(11.0)
    arr_gap = Inches(0.3)

    # Row 1: IDD images -> Binary masks
    add_box(s11, x_c1, Inches(1.7), fc_w6, fc_h6, "IDD Images", "10,003 Scenes", style="green")
    add_right_arrow(s11, x_c1 + fc_w6 + Inches(0.05), Inches(1.7) + fc_h6 / 2, length=arr_gap)
    add_box(s11, x_c2, Inches(1.7), fc_w6, fc_h6, "Binary Masks", "Collapsed 41->2", style="gold")

    # Down to Teacher
    add_down_arrow(s11, x_c2 + fc_w6 / 2, Inches(1.7) + fc_h6, gap_y)
    add_box(s11, x_c2, Inches(1.7) + fc_h6 + gap_y, fc_w6, fc_h6, "Train Teacher", "MobileNetV3-Large", style="green")

    # Row 3: Own data (CVAT) -> Fine-tune
    y_row3 = Inches(1.7) + 2 * (fc_h6 + gap_y)
    add_box(s11, x_c1, y_row3, fc_w6, fc_h6, "Own Data (CVAT)", "Night, Terrain, Cattle", style="green")
    add_right_arrow(s11, x_c1 + fc_w6 + Inches(0.05), y_row3 + fc_h6 / 2, length=arr_gap)
    add_box(s11, x_c2, y_row3, fc_w6, fc_h6, "Fine-Tune", "Domain Adaptation", style="gold")

    # Down to Held-out test
    y_row4 = y_row3 + fc_h6 + gap_y
    add_down_arrow(s11, x_c2 + fc_w6 / 2, y_row3 + fc_h6, gap_y)
    add_box(s11, x_c2, y_row4, fc_w6, fc_h6, "Held-Out Test", "Unbiased Evaluation", style="dark")

    # -------------------------------------------------------------
    # SLIDE 12: Progress and Next Steps
    # -------------------------------------------------------------
    s12 = add_base_slide(12)
    add_title(s12, "Progress and Next Steps")
    b12 = s12.shapes.add_textbox(Inches(1.35), Inches(1.65), Inches(6.8), Inches(5.2))
    tf12 = b12.text_frame
    tf12.word_wrap = True

    prog_points = [
        ("Completed Milestones:", " End-to-end inference pipeline architecture with automated per-stage timing and regression smoke tests. Integrated YOLOv8n dynamic obstacle detection with dilated corridor subtraction and coupled steering/velocity decision logic. Self-calibrating IPM with IMU telemetry and dynamic horizon tracking."),
        ("Current Limitations:", " FusionNet operates on calibrated heuristic priors pending full multi-condition terrain annotation. Evidential uncertainty calibration requires expanded out-of-distribution validation across severe monsoon weather."),
        ("Strategic Next Steps:", " Distributed GPU knowledge distillation from Teacher to Student (T=2.0). Post-training INT8 quantization and structured pruning for ultra-low latency edge deployment. Physical vehicle hardware-in-the-loop (HIL) microcontroller demonstration."),
    ]
    for i, (lead, body) in enumerate(prog_points):
        p = tf12.paragraphs[0] if i == 0 else tf12.add_paragraph()
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = "• " + lead + "\n"
        r1.font.name = FONT_BODY
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_LEAD
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_BODY
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_BODY

    # Flowchart on right (5 boxes)
    y_start = Inches(1.65)
    add_box(s12, fc_x, y_start, fc_w, fc_h5, "Data Preparation", "Dataset curation & binary mask generation", style="green")
    add_down_arrow(s12, fc_x + fc_w / 2, y_start + fc_h5, gap5)
    y_start += fc_h5 + gap5
    add_box(s12, fc_x, y_start, fc_w, fc_h5, "Train Teacher", "MobileNetV3-Large baseline model", style="green")
    add_down_arrow(s12, fc_x + fc_w / 2, y_start + fc_h5, gap5)
    y_start += fc_h5 + gap5
    add_box(s12, fc_x, y_start, fc_w, fc_h5, "Train FusionNet", "Multi-signal trust fusion optimization", style="gold")
    add_down_arrow(s12, fc_x + fc_w / 2, y_start + fc_h5, gap5)
    y_start += fc_h5 + gap5
    add_box(s12, fc_x, y_start, fc_w, fc_h5, "Distill + Compress", "Student distillation & INT8 quantization", style="green")
    add_down_arrow(s12, fc_x + fc_w / 2, y_start + fc_h5, gap5)
    y_start += fc_h5 + gap5
    add_box(s12, fc_x, y_start, fc_w, fc_h5, "Hardware Demo", "Embedded vehicle deployment & validation", style="dark")

    # -------------------------------------------------------------
    # SLIDE 13: References
    # -------------------------------------------------------------
    s13 = add_base_slide(13)
    add_title(s13, "References")
    b13 = s13.shapes.add_textbox(Inches(1.35), Inches(1.65), Inches(8.5), Inches(5.2))
    tf13 = b13.text_frame
    tf13.word_wrap = True

    refs = [
        ("1. ", "Varma, G., Subramanian, A., Namboodiri, V., Chandraker, M., & Jawahar, C. V. (2019). IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments. In ", "IEEE Winter Conference on Applications of Computer Vision (WACV)", "."),
        ("2. ", "Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for MobileNetV3. In ", "IEEE/CVF International Conference on Computer Vision (ICCV)", "."),
        ("3. ", "Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In ", "Medical Image Computing and Computer-Assisted Intervention (MICCAI)", "."),
        ("4. ", "Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential Deep Learning to Quantify Classification Uncertainty. In ", "Advances in Neural Information Processing Systems (NeurIPS)", "."),
        ("5. ", "Rasib, M. N., Javid, M. A., & Kim, D. (2021). Drivable road region detection and steering angle estimation for autonomous driving in unstructured environments. ", "IEEE Access", ", 9, 154210–154222."),
    ]
    for i, (num, authors_title, venue, ending) in enumerate(refs):
        p = tf13.paragraphs[0] if i == 0 else tf13.add_paragraph()
        p.space_after = Pt(12)
        r0 = p.add_run()
        r0.text = num
        r0.font.name = FONT_BODY
        r0.font.bold = True
        r0.font.size = Pt(12)
        r0.font.color.rgb = COLOR_LEAD

        r1 = p.add_run()
        r1.text = authors_title
        r1.font.name = FONT_BODY
        r1.font.size = Pt(11.5)
        r1.font.color.rgb = COLOR_BODY

        r2 = p.add_run()
        r2.text = venue
        r2.font.name = FONT_BODY
        r2.font.size = Pt(11.5)
        r2.font.italic = True
        r2.font.color.rgb = COLOR_SUBTITLE

        r3 = p.add_run()
        r3.text = ending
        r3.font.name = FONT_BODY
        r3.font.size = Pt(11.5)
        r3.font.color.rgb = COLOR_BODY

    # -------------------------------------------------------------
    # SLIDE 14: Thank You
    # -------------------------------------------------------------
    s14 = add_base_slide(14)
    b14 = s14.shapes.add_textbox(Inches(3.5), Inches(2.2), Inches(6.5), Inches(3.5))
    tf14 = b14.text_frame
    tf14.word_wrap = True

    p_ty = tf14.paragraphs[0]
    p_ty.alignment = PP_ALIGN.CENTER
    p_ty.text = "Thank You"
    p_ty.font.name = FONT_TITLE
    p_ty.font.size = Pt(44)
    p_ty.font.bold = True
    p_ty.font.color.rgb = COLOR_TITLE

    p_qa = tf14.add_paragraph()
    p_qa.alignment = PP_ALIGN.CENTER
    p_qa.text = "Questions & Technical Discussion"
    p_qa.font.name = FONT_BODY
    p_qa.font.size = Pt(18)
    p_qa.font.italic = True
    p_qa.font.color.rgb = COLOR_SUBTITLE
    p_qa.space_before = Pt(12)

    p_dt = tf14.add_paragraph()
    p_dt.alignment = PP_ALIGN.CENTER
    p_dt.text = "BACSE324 · Embedded Systems Architecture"
    p_dt.font.name = FONT_BODY
    p_dt.font.bold = True
    p_dt.font.size = Pt(14)
    p_dt.font.color.rgb = COLOR_BODY
    p_dt.space_before = Pt(24)

    p_names = tf14.add_paragraph()
    p_names.alignment = PP_ALIGN.CENTER
    p_names.text = "Sunil Kumar (25BCE0388)   |   Varshith R S (25BCE0351)   |   Suyash Gupta (25BCE0371)"
    p_names.font.name = FONT_BODY
    p_names.font.size = Pt(12.5)
    p_names.font.color.rgb = COLOR_LEAD
    p_names.space_before = Pt(6)

    # Save presentation
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"Successfully generated presentation at {output_path}")

if __name__ == "__main__":
    assets = r"C:\Users\varsh\.gemini\antigravity\brain\74843c63-b7db-4812-b3c5-7b92e0eceb31\slide_assets"
    out = r"d:\Projects\ESA 4\ATWFS\outputs\presentation\ATWFS_Presentation.pptx"
    create_presentation(assets, out)
