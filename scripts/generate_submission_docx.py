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
    repo_url="https://github.com/sakshamgupta/autoregressive-text-generation-fgai",
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
        "and the vocabulary is fitted solely on the training tokens. Input sequences of length 8 predict the 9th token."
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
        ("Simple RNN", "985,022", "3", "5.0287", "152.73", "18.10%", "5.1567", "17.60%", "38.41%"),
        ("Bidirectional RNN", "1,513,406", "2", "5.1401", "170.74", "16.88%", "5.2474", "16.94%", "35.96%"),
        ("LSTM", "1,207,358", "4", "4.9778", "145.16", "18.70%", "5.1261", "19.68%", "39.17%"),
        ("GRU", "1,133,246", "4", "4.9701", "144.04", "18.92%", "5.1361", "19.49%", "38.95%"),
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
    r_h3 = h3.add_run("3. Cross-Model Temperature Analysis (Averaged Over Seeds)")
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
        ("T = 0.2", "0.507", "0.701", "18.7%", "4–6 words", "Grammatically strict; repetitive phrasing traps"),
        ("T = 0.5", "0.679", "0.878", "5.0%", "0–4 words", "Optimal empirical trade-off: high diversity with fluent phrasing"),
        ("T = 1.0", "0.855", "1.000", "0.0%", "0 words", "High lexical richness; slight syntactic drift across clauses"),
        ("T = 1.5", "0.905", "1.000", "0.0%", "0 words", "Excessive entropy; random token sequence, ungrammatical"),
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
        "• LSTM and GRU perform competitively as the top two architectures: GRU achieves the lowest validation loss (4.9701, PPL 144.04), "
        "while LSTM achieves the highest unseen test-set top-1 accuracy (19.68%) and top-5 accuracy (39.17%).\n"
        "• Bidirectional RNN: By properly concatenating the forward final state h_n[-2] and backward final state h_n[-1], BiRNN captures the complete sequence context. "
        "However, with 1,513,406 parameters it exhibits earlier overfitting (peaking at epoch 2) compared to unidirectional models.\n"
        "• Sampling Strategy: T = 0.5 provides the best empirical balance between lexical diversity (Distinct-1: ~0.68) and low phrase repetition (trigram repetition under 5%)."
    )
    
    output_path = os.path.join(os.getcwd(), output_filename)
    doc.save(output_path)
    print(f"Generated Word document at: {output_path}")
    return output_path


def build_submission_pdf(
    student_name="Saksham Gupta",
    reg_num="230911186",
    repo_url="https://github.com/sakshamgupta/autoregressive-text-generation-fgai",
    output_filename="SakshamGupta_230911186_FGAI_Assignment2.pdf"
):
    pdf_path = os.path.join(os.getcwd(), output_filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=15, leading=18, textColor=colors.HexColor('#1e3c72'), alignment=1, spaceAfter=4)
    sub_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontSize=10, leading=14, fontName='Helvetica-Oblique', alignment=1, spaceAfter=14)
    h1_style = ParagraphStyle('SecH1', parent=styles['Heading2'], fontSize=12, leading=15, textColor=colors.HexColor('#1e3c72'), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=8.5, leading=12, spaceAfter=6)
    table_text = ParagraphStyle('TableT', parent=styles['Normal'], fontSize=8, leading=10)
    table_head = ParagraphStyle('TableH', parent=styles['Normal'], fontSize=8, leading=10, fontName='Helvetica-Bold')

    story = []
    story.append(Paragraph("FOUNDATIONS OF GENERATIVE AI (ICT4403)", title_style))
    story.append(Paragraph("Assignment 2: Autoregressive Text Generation System<br/>Comparative Study of Simple RNN, Bidirectional RNN, LSTM, and GRU", sub_style))

    # Meta Table
    meta_table_data = [
        [Paragraph("<b>Student Name</b>", table_text), Paragraph(student_name, table_text)],
        [Paragraph("<b>Registration Number</b>", table_text), Paragraph(reg_num, table_text)],
        [Paragraph("<b>Assignment Title</b>", table_text), Paragraph("Assignment 2 - Autoregressive Text Generation System", table_text)],
        [Paragraph("<b>System Title</b>", table_text), Paragraph("Comparative Word-Level Autoregressive Text Generation on Lewis Carroll's Alice in Wonderland using Simple RNN, Bidirectional RNN, LSTM and GRU", table_text)],
        [Paragraph("<b>Dataset & Source</b>", table_text), Paragraph("Alice's Adventures in Wonderland (Lewis Carroll)<br/>https://www.gutenberg.org/files/11/11-0.txt", table_text)],
        [Paragraph("<b>GitHub Repository</b>", table_text), Paragraph(f'<font color="#0066cc"><u>{repo_url}</u></font>', table_text)],
    ]
    meta_table = Table(meta_table_data, colWidths=[140, 390])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Section 1
    story.append(Paragraph("1. System Overview & Problem Statement", h1_style))
    story.append(Paragraph(
        "This project implements and evaluates an autoregressive next-word language generation system. "
        "Four distinct recurrent neural network architectures were built in PyTorch and evaluated under strictly comparable conditions "
        "on Lewis Carroll's <i>Alice's Adventures in Wonderland</i>: Simple RNN, Bidirectional RNN, LSTM, and GRU.<br/>"
        "To eliminate data leakage, the corpus is split chronologically into 80% Train, 10% Validation, and 10% Test blocks, "
        "and the vocabulary is fitted solely on the training tokens. Input sequences of length 8 predict the succeeding token.",
        body_style
    ))

    # Section 2
    story.append(Paragraph("2. Quantitative Evaluation Summary (Best Epoch & Test Set)", h1_style))
    q_data = [
        [Paragraph(h, table_head) for h in ["Architecture", "Parameters", "Best Ep", "Val Loss", "Val PPL", "Val Top-1", "Test Loss", "Test Top-1", "Test Top-5"]],
        [Paragraph("<b>Simple RNN</b>", table_text), Paragraph("985,022", table_text), Paragraph("3", table_text), Paragraph("5.0287", table_text), Paragraph("152.73", table_text), Paragraph("18.10%", table_text), Paragraph("5.1567", table_text), Paragraph("17.60%", table_text), Paragraph("38.41%", table_text)],
        [Paragraph("<b>BiRNN</b>", table_text), Paragraph("1,513,406", table_text), Paragraph("2", table_text), Paragraph("5.1401", table_text), Paragraph("170.74", table_text), Paragraph("16.88%", table_text), Paragraph("5.2474", table_text), Paragraph("16.94%", table_text), Paragraph("35.96%", table_text)],
        [Paragraph("<b>LSTM</b>", table_text), Paragraph("1,207,358", table_text), Paragraph("4", table_text), Paragraph("4.9778", table_text), Paragraph("145.16", table_text), Paragraph("18.70%", table_text), Paragraph("5.1261", table_text), Paragraph("<b>19.68%</b>", table_text), Paragraph("<b>39.17%</b>", table_text)],
        [Paragraph("<b>GRU</b>", table_text), Paragraph("1,133,246", table_text), Paragraph("4", table_text), Paragraph("<b>4.9701</b>", table_text), Paragraph("<b>144.04</b>", table_text), Paragraph("<b>18.92%</b>", table_text), Paragraph("5.1361", table_text), Paragraph("19.49%", table_text), Paragraph("38.95%", table_text)],
    ]
    q_table = Table(q_data, colWidths=[80, 65, 45, 55, 55, 55, 55, 60, 60])
    q_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e9ecef')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(q_table)
    story.append(Spacer(1, 10))

    # Section 3
    story.append(Paragraph("3. Cross-Model Temperature Analysis (Averaged Over Seeds)", h1_style))
    t_data = [
        [Paragraph(h, table_head) for h in ["Temperature", "Distinct-1", "Distinct-2", "Trigram Repetition", "Longest Loop", "Empirical Quality"]],
        [Paragraph("<b>T = 0.2</b>", table_text), Paragraph("0.507", table_text), Paragraph("0.701", table_text), Paragraph("18.7%", table_text), Paragraph("4–6 words", table_text), Paragraph("Grammatically strict; repetitive phrasing loops", table_text)],
        [Paragraph("<b>T = 0.5</b>", table_text), Paragraph("0.679", table_text), Paragraph("0.878", table_text), Paragraph("5.0%", table_text), Paragraph("0–4 words", table_text), Paragraph("Optimal empirical trade-off: high diversity with fluent phrasing", table_text)],
        [Paragraph("<b>T = 1.0</b>", table_text), Paragraph("0.855", table_text), Paragraph("1.000", table_text), Paragraph("0.0%", table_text), Paragraph("0 words", table_text), Paragraph("High lexical richness; slight syntactic drift across clauses", table_text)],
        [Paragraph("<b>T = 1.5</b>", table_text), Paragraph("0.905", table_text), Paragraph("1.000", table_text), Paragraph("0.0%", table_text), Paragraph("0 words", table_text), Paragraph("High entropy; ungrammatical, random token sequences", table_text)],
    ]
    t_table = Table(t_data, colWidths=[65, 55, 55, 75, 60, 220])
    t_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e9ecef')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_table)
    story.append(Spacer(1, 10))

    # Section 4
    story.append(Paragraph("4. Conclusion & Model Comparison", h1_style))
    story.append(Paragraph(
        "• <b>LSTM and GRU</b> perform competitively as the top two architectures: GRU achieves the lowest validation loss (4.9701, PPL 144.04), "
        "while LSTM edges ahead on unseen test-set top-1 accuracy (19.68%) and top-5 accuracy (39.17%).<br/>"
        "• <b>Bidirectional RNN:</b> By properly concatenating the forward final state h_n[-2] and backward final state h_n[-1], BiRNN captures the complete sequence context. "
        "However, with 1,513,406 parameters it exhibits earlier overfitting (peaking at epoch 2) compared to unidirectional models.<br/>"
        "• <b>Sampling Strategy:</b> T = 0.5 provides the best empirical balance between lexical diversity (Distinct-1: ~0.68) and low phrase repetition (trigram repetition under 5%).",
        body_style
    ))

    doc.build(story)
    print(f"Generated PDF document at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    docx_file = build_submission_docx()
    pdf_file = build_submission_pdf()
