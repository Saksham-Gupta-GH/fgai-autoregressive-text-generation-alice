import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def build_submission_docx(
    student_name="Saksham Gupta",
    reg_num="230911186",
    repo_url="https://github.com/Saksham-Gupta-GH/fgai-autoregressive-text-generation-alice",
    output_filename="SakshamGupta_230911186_FGAI_Assignment2.docx"
):
    doc = Document()
    
    # Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title.add_run("FOUNDATIONS OF GENERATIVE AI (ICT4403)")
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(30, 60, 114)
    
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = sub.add_run("Assignment 2: Autoregressive Text Generation System\nComparative Study of Simple RNN, Bidirectional RNN, LSTM, and GRU")
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Metadata Table
    table = doc.add_table(rows=6, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    meta_data = [
        ("Student Name", student_name),
        ("Registration Number", reg_num),
        ("Assignment Title", "Assignment 2 - Autoregressive Text Generation System"),
        ("System Title", "Comparative Word-Level Autoregressive Text Generation on Lewis Carroll's Alice in Wonderland using Simple RNN, Bidirectional RNN, LSTM and GRU"),
        ("Dataset Name & Source", "Alice's Adventures in Wonderland (Lewis Carroll)\nhttps://www.gutenberg.org/files/11/11-0.txt"),
        ("GitHub Repository Link", repo_url)
    ]
    
    for i, (k, v) in enumerate(meta_data):
        row = table.rows[i]
        c0 = row.cells[0]
        c1 = row.cells[1]
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(10)
        
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.size = Pt(10)
        if "http" in v:
            r1.font.color.rgb = RGBColor(0, 102, 204)
            r1.font.underline = True
            
    doc.add_paragraph().paragraph_format.space_after = Pt(16)
    
    # Section 1: Overview
    h1 = doc.add_heading(level=1)
    r_h1 = h1.add_run("1. System Overview & Problem Statement")
    r_h1.font.size = Pt(13)
    r_h1.font.bold = True
    r_h1.font.color.rgb = RGBColor(30, 60, 114)
    
    doc.add_paragraph(
        "This project implements, evaluates, and analyzes an autoregressive next-word language generation system. "
        "Four recurrent neural network architectures were implemented in PyTorch and evaluated under strictly comparable "
        "experimental conditions on Lewis Carroll's 'Alice's Adventures in Wonderland' corpus:\n"
        "• Simple RNN (Vanilla Elman recurrent network)\n"
        "• Bidirectional RNN (Concatenating forward final state h_n[-2] and backward final state h_n[-1])\n"
        "• LSTM (Long Short-Term Memory with forget, input, and output gates)\n"
        "• GRU (Gated Recurrent Unit with reset and update gates)\n\n"
        "To eliminate data leakage, the corpus is split chronologically into 80% Train, 10% Validation, and 10% Test blocks, "
        "and the vocabulary is fitted solely on the training tokens (min_freq=2). Input sequences of length 6 predict the 7th token."
    )
    
    # Section 2: Quantitative Results
    h2 = doc.add_heading(level=1)
    r_h2 = h2.add_run("2. Quantitative Evaluation Summary (Best Epoch & Test Set)")
    r_h2.font.size = Pt(13)
    r_h2.font.bold = True
    r_h2.font.color.rgb = RGBColor(30, 60, 114)
    
    q_table = doc.add_table(rows=5, cols=9)
    q_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    q_table.autofit = True
    
    q_headers = ["Architecture", "Parameters", "Best Ep", "Val Loss", "Val PPL", "Val Top-1", "Test Loss", "Test Top-1", "Test Top-5"]
    for j, h in enumerate(q_headers):
        cell = q_table.rows[0].cells[j]
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.font.bold = True
        run.font.size = Pt(8.5)
        
    q_rows = [
        ("Simple RNN", "798,484", "4", "4.0741", "58.80", "19.73%", "4.1584", "20.80%", "45.28%"),
        ("Bidirectional RNN", "1,262,868", "2", "4.2804", "72.27", "18.66%", "4.3192", "18.47%", "42.26%"),
        ("LSTM", "1,193,236", "5", "3.9903", "54.07", "21.87%", "4.0576", "23.29%", "47.23%"),
        ("GRU", "1,061,652", "3", "4.0353", "56.56", "20.55%", "4.0900", "22.53%", "46.51%"),
    ]
    
    for i, row_data in enumerate(q_rows):
        for j, val in enumerate(row_data):
            cell = q_table.rows[i+1].cells[j]
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.size = Pt(8.5)
            if j == 0:
                run.font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(14)
    
    # Section 3: Qualitative & Temperature Analysis
    h3 = doc.add_heading(level=1)
    r_h3 = h3.add_run("3. Cross-Model Temperature Analysis (Averaged Over Models and Seed Prompts, n = 3 per cell)")
    r_h3.font.size = Pt(13)
    r_h3.font.bold = True
    r_h3.font.color.rgb = RGBColor(30, 60, 114)
    
    t_table = doc.add_table(rows=5, cols=6)
    t_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_table.autofit = True
    
    t_headers = ["Temperature (T)", "Distinct-1", "Distinct-2", "Trigram Repetition", "Longest Loop", "Empirical Quality"]
    for j, h in enumerate(t_headers):
        cell = t_table.rows[0].cells[j]
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.font.bold = True
        run.font.size = Pt(8.5)
        
    t_rows = [
        ("T = 0.2", "0.517", "0.674", "23.7%", "0–6 words", "Strict syntax; prone to repetitive phrase traps (GRU 52.5%)"),
        ("T = 0.5", "0.678", "0.900", "3.0%", "0 words", "Optimal empirical trade-off: high diversity with fluent dialogue; LSTM and GRU comparable"),
        ("T = 1.0", "0.855", "0.992", "0.0%", "0 words", "High lexical variety; slight syntactic drift across clauses"),
        ("T = 1.5", "0.916", "1.000", "0.0%", "0 words", "High entropy; syntactic collapse into ungrammatical token salad"),
    ]
    
    for i, row_data in enumerate(t_rows):
        for j, val in enumerate(row_data):
            cell = t_table.rows[i+1].cells[j]
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.size = Pt(8.5)
            if j == 0:
                run.font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Section 4: Conclusion
    h4 = doc.add_heading(level=1)
    r_h4 = h4.add_run("4. Conclusion & Model Comparison")
    r_h4.font.size = Pt(13)
    r_h4.font.bold = True
    r_h4.font.color.rgb = RGBColor(30, 60, 114)
    
    doc.add_paragraph(
        "• LSTM edges ahead across next-word prediction metrics: lowest validation loss (3.9903, PPL 54.07), "
        "highest validation top-1 accuracy (21.87%), and highest held-out test top-1 accuracy (23.29%) and top-5 accuracy (47.23%).\n"
        "• GRU performs closely behind in second place (Val Loss 4.0353, Test Top-1 22.53%) with 11% fewer parameters (1,061,652 vs. 1,193,236). Simple RNN remains competitive in loss (4.0741, PPL 58.80).\n"
        "• For sequential generation, LSTM and GRU are comparable: GRU achieves 0.0% trigram repetition at T=0.5, while LSTM produces the most fluent story phrasing in our samples.\n"
        "• Statistical Evaluation & Single-Seed Limitation: While all models were trained with SEED=42 to enforce strictly identical conditions, the 0.045 cross-entropy delta between LSTM and GRU falls within typical single-run variance. Multi-seed runs (3–5 seeds) with mean ± standard deviation reporting are recommended for production-scale validation.\n"
        "• Bidirectional RNN: By concatenating forward final state h_n[-2] and backward final state h_n[-1], BiRNN correctly encodes the complete input context. "
        "However, with 1,262,868 parameters it exhibits earlier overfitting (peaking at epoch 2) compared to causal unidirectional models.\n"
        "• Sampling Strategy: T = 0.5 provides the optimal empirical balance between lexical diversity (Distinct-1: 0.60–0.73) and phrase stability, achieving low trigram repetition across models (LSTM: 5.0%, GRU: 0.0%, BiRNN: 1.0%, Simple RNN: 6.1%)."
    )
    
    output_path = os.path.join(os.getcwd(), output_filename)
    doc.save(output_path)
    print(f"Generated Word document at: {output_path}")
    return output_path


def build_submission_pdf(
    student_name="Saksham Gupta",
    reg_num="230911186",
    repo_url="https://github.com/Saksham-Gupta-GH/fgai-autoregressive-text-generation-alice",
    output_filename="SakshamGupta_230911186_FGAI_Assignment2.pdf"
):
    pdf_path = os.path.join(os.getcwd(), output_filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=24, bottomMargin=24)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=13, leading=16, textColor=colors.HexColor('#1e3c72'), alignment=1, spaceAfter=2)
    sub_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontSize=9, leading=12, fontName='Helvetica-Oblique', alignment=1, spaceAfter=6)
    h1_style = ParagraphStyle('SecH1', parent=styles['Heading2'], fontSize=10, leading=13, textColor=colors.HexColor('#1e3c72'), spaceBefore=5, spaceAfter=3)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=7.5, leading=10, spaceAfter=4)
    table_text = ParagraphStyle('TableT', parent=styles['Normal'], fontSize=7.5, leading=9.5)
    table_head = ParagraphStyle('TableH', parent=styles['Normal'], fontSize=7.5, leading=9.5, fontName='Helvetica-Bold')

    story = []
    story.append(Paragraph("FOUNDATIONS OF GENERATIVE AI (ICT4403)", title_style))
    story.append(Paragraph("Assignment 2: Autoregressive Text Generation System — Comparative Study", sub_style))

    # Meta Table
    meta_table_data = [
        [Paragraph("<b>Student Name</b>", table_text), Paragraph(student_name, table_text)],
        [Paragraph("<b>Registration Number</b>", table_text), Paragraph(reg_num, table_text)],
        [Paragraph("<b>Assignment Title</b>", table_text), Paragraph("Assignment 2 - Autoregressive Text Generation System", table_text)],
        [Paragraph("<b>System Title</b>", table_text), Paragraph("Comparative Word-Level Autoregressive Text Generation on Lewis Carroll's Alice in Wonderland using Simple RNN, BiRNN, LSTM, GRU", table_text)],
        [Paragraph("<b>Dataset & Source</b>", table_text), Paragraph("Alice's Adventures in Wonderland (Lewis Carroll) | https://www.gutenberg.org/files/11/11-0.txt", table_text)],
        [Paragraph("<b>GitHub Repository</b>", table_text), Paragraph(f'<font color="#0066cc"><u>{repo_url}</u></font>', table_text)],
    ]
    meta_table = Table(meta_table_data, colWidths=[120, 420])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(meta_table)

    # Section 1
    story.append(Paragraph("1. System Overview & Problem Statement", h1_style))
    story.append(Paragraph(
        "Four recurrent neural language models (Simple RNN, BiRNN with forward-backward final state concatenation [h_n[-2], h_n[-1]], LSTM, and GRU) were trained under strictly controlled conditions on Alice's Adventures in Wonderland. "
        "To eliminate data leakage, the corpus is partitioned chronologically (80% Train, 10% Val, 10% Test) and the vocabulary (1,300 tokens, min_freq=2) is fitted exclusively on training tokens. Context sequence length is 6 tokens.",
        body_style
    ))

    # Section 2
    story.append(Paragraph("2. Quantitative Evaluation Summary (Best Epoch & Test Set)", h1_style))
    q_data = [
        [Paragraph(h, table_head) for h in ["Architecture", "Parameters", "Best Ep", "Val Loss", "Val PPL", "Val Top-1", "Test Loss", "Test Top-1", "Test Top-5"]],
        [Paragraph("<b>Simple RNN</b>", table_text), Paragraph("798,484", table_text), Paragraph("4", table_text), Paragraph("4.0741", table_text), Paragraph("58.80", table_text), Paragraph("19.73%", table_text), Paragraph("4.1584", table_text), Paragraph("20.80%", table_text), Paragraph("45.28%", table_text)],
        [Paragraph("<b>BiRNN</b>", table_text), Paragraph("1,262,868", table_text), Paragraph("2", table_text), Paragraph("4.2804", table_text), Paragraph("72.27", table_text), Paragraph("18.66%", table_text), Paragraph("4.3192", table_text), Paragraph("18.47%", table_text), Paragraph("42.26%", table_text)],
        [Paragraph("<b>LSTM</b>", table_text), Paragraph("1,193,236", table_text), Paragraph("5", table_text), Paragraph("<b>3.9903</b>", table_text), Paragraph("<b>54.07</b>", table_text), Paragraph("<b>21.87%</b>", table_text), Paragraph("<b>4.0576</b>", table_text), Paragraph("<b>23.29%</b>", table_text), Paragraph("<b>47.23%</b>", table_text)],
        [Paragraph("<b>GRU</b>", table_text), Paragraph("1,061,652", table_text), Paragraph("3", table_text), Paragraph("4.0353", table_text), Paragraph("56.56", table_text), Paragraph("20.55%", table_text), Paragraph("4.0900", table_text), Paragraph("22.53%", table_text), Paragraph("46.51%", table_text)],
    ]
    q_table = Table(q_data, colWidths=[75, 65, 42, 50, 50, 52, 50, 56, 56])
    q_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e9ecef')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(q_table)

    # Section 3
    story.append(Paragraph("3. Cross-Model Temperature Analysis (Averaged Over Models and Seed Prompts, n = 3 per cell)", h1_style))
    t_data = [
        [Paragraph(h, table_head) for h in ["Temperature", "Distinct-1", "Distinct-2", "Trigram Repetition", "Longest Loop", "Empirical Quality Observation"]],
        [Paragraph("<b>T = 0.2</b>", table_text), Paragraph("0.517", table_text), Paragraph("0.674", table_text), Paragraph("23.7%", table_text), Paragraph("0–6 words", table_text), Paragraph("Strict syntax; repetitive looping traps across models (GRU 52.5%)", table_text)],
        [Paragraph("<b>T = 0.5</b>", table_text), Paragraph("0.678", table_text), Paragraph("0.900", table_text), Paragraph("3.0%", table_text), Paragraph("0 words", table_text), Paragraph("Optimal empirical trade-off: high diversity with fluent story dialogue; LSTM and GRU comparable", table_text)],
        [Paragraph("<b>T = 1.0</b>", table_text), Paragraph("0.855", table_text), Paragraph("0.992", table_text), Paragraph("0.0%", table_text), Paragraph("0 words", table_text), Paragraph("High lexical richness; slight syntactic drift across longer clauses", table_text)],
        [Paragraph("<b>T = 1.5</b>", table_text), Paragraph("0.916", table_text), Paragraph("1.000", table_text), Paragraph("0.0%", table_text), Paragraph("0 words", table_text), Paragraph("High entropy; syntactic collapse into ungrammatical token sequences", table_text)],
    ]
    t_table = Table(t_data, colWidths=[55, 52, 52, 70, 55, 256])
    t_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e9ecef')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_table)

    # Section 4
    story.append(Paragraph("4. Conclusion & Model Comparison", h1_style))
    story.append(Paragraph(
        "• <b>LSTM and GRU Comparison:</b> LSTM edges ahead across prediction metrics (Val Loss 3.9903, PPL 54.07, Test Top-1 23.29%), with GRU performing closely behind (Val Loss 4.0353, Test Top-1 22.53%) while requiring 11% fewer parameters. Simple RNN remains competitive in loss (4.0741). For text generation, LSTM and GRU are comparable: GRU achieves 0.0% trigram repetition at T=0.5, while LSTM produces the most fluent story phrasing in our samples.<br/>"
        "• <b>Single-Seed Limitation:</b> Experiments were run under SEED=42 for strict controlled comparison; the 0.045 loss delta between LSTM and GRU falls within typical single-run variance margins. Evaluating across 3–5 seeds with mean ± std is recommended for production-scale validation.<br/>"
        "• <b>Bidirectional RNN:</b> Concatenating forward state h_n[-2] and backward state h_n[-1] feeds complete context to the classifier, but its 1,262,868 parameters overfit by epoch 2 (val loss 4.2804).<br/>"
        "• <b>Temperature Sampling:</b> T = 0.5 provides the optimal trade-off between lexical diversity (Distinct-1: 0.60–0.73) and phrase stability, achieving low repetition across models (LSTM 5.0%, GRU 0.0%, BiRNN 1.0%, Simple RNN 6.1%).",
        body_style
    ))

    doc.build(story)
    print(f"Generated PDF document at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    docx_file = build_submission_docx()
    pdf_file = build_submission_pdf()
