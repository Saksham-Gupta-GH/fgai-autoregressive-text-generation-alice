import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def add_docx_hyperlink(paragraph, url, text, color="0066CC", underline=True):
    """
    Inserts a genuine clickable hyperlink into a python-docx paragraph.
    """
    part = paragraph.part
    r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)
    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    if color:
        c = OxmlElement('w:color')
        c.set(qn('w:val'), color)
        rPr.append(c)
    if underline:
        u = OxmlElement('w:u')
        u.set(qn('w:val'), 'single')
        rPr.append(u)
    new_run.append(rPr)
    new_run.text = text
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink


def build_submission_docx(
    student_name="Saksham Gupta",
    reg_num="230911186",
    repo_url="https://github.com/Saksham-Gupta-GH/fgai-autoregressive-text-generation-alice",
    output_filename="SakshamGupta_230911186_FGAI_Assignment2.docx"
):
    doc = Document()
    
    # Margins
    for section in doc.sections:
        section.top_margin = Inches(1.2)
        section.bottom_margin = Inches(1.2)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title.add_run("Foundations of Generative AI (ICT4403)")
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0, 0, 0)
    
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = sub.add_run("Assignment 2 – Submission Details")
    run_sub.font.size = Pt(13)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(20)
    
    # Metadata Table (7 Rows matching friend's structure)
    table = doc.add_table(rows=7, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    meta_rows = [
        ("Student name", student_name, None),
        ("Registration number", reg_num, None),
        ("Assignment title", "Foundations of Generative AI (ICT4403) – Assignment 2", None),
        ("Title of the developed autoregressive text-generation system", "Comparative Word-Level Autoregressive Text Generation on Lewis Carroll's Alice's Adventures in Wonderland using Simple RNN, Bidirectional RNN, LSTM, and GRU", None),
        ("Dataset name", "Alice's Adventures in Wonderland by Lewis Carroll (Project Gutenberg eBook #11)", None),
        ("Dataset link", "", "https://www.gutenberg.org/files/11/11-0.txt"),
        ("GitHub repository link", "", repo_url)
    ]
    
    for i, (k, text_prefix, link_url) in enumerate(meta_rows):
        row = table.rows[i]
        c0 = row.cells[0]
        c1 = row.cells[1]
        c0.width = Inches(2.6)
        c1.width = Inches(4.2)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(5)
        p0.paragraph_format.space_after = Pt(5)
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(10.5)
        
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_before = Pt(5)
        p1.paragraph_format.space_after = Pt(5)
        if text_prefix:
            r1 = p1.add_run(text_prefix)
            r1.font.size = Pt(10.5)
        if link_url:
            add_docx_hyperlink(p1, link_url, link_url)
            
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
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=13, leading=16, textColor=colors.black, alignment=1, spaceAfter=2)
    sub_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontSize=9, leading=12, fontName='Helvetica-Oblique', alignment=1, spaceAfter=6)
    h1_style = ParagraphStyle('SecH1', parent=styles['Heading2'], fontSize=10, leading=13, textColor=colors.black, spaceBefore=5, spaceAfter=3)
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
        [Paragraph("<b>Dataset & Source</b>", table_text), Paragraph('Alice\'s Adventures in Wonderland (Lewis Carroll) | <a href="https://www.gutenberg.org/files/11/11-0.txt"><font color="#0066cc"><u>https://www.gutenberg.org/files/11/11-0.txt</u></font></a>', table_text)],
        [Paragraph("<b>GitHub Repository</b>", table_text), Paragraph(f'<a href="{repo_url}"><font color="#0066cc"><u>{repo_url}</u></font></a>', table_text)],
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
        "Four recurrent neural language models (Simple RNN, BiRNN with forward-backward final state concatenation [h_n[-2], h_n[-1]], LSTM, and GRU) were trained "
        "under strictly controlled conditions on Alice's Adventures in Wonderland. To eliminate data leakage, the corpus is partitioned chronologically (80% Train, 10% "
        "Val, 10% Test) and the vocabulary (1,300 tokens, min_freq=2) is fitted exclusively on training tokens. Context sequence length is 6 tokens. Held-out test set has an 11.17% &lt;unk&gt; token rate.",
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
        [Paragraph(h, table_head) for h in ["T", "Distinct-1", "Distinct-2", "Trigram Repetition", "Longest Loop", "Empirical Quality Observation"]],
        [Paragraph("<b>0.2</b>", table_text), Paragraph("0.517", table_text), Paragraph("0.674", table_text), Paragraph("23.7%", table_text), Paragraph("0–6 words", table_text), Paragraph("Strict syntax; repetitive looping traps across models (GRU 52.5%)", table_text)],
        [Paragraph("<b>0.5</b>", table_text), Paragraph("0.678", table_text), Paragraph("0.900", table_text), Paragraph("3.0%", table_text), Paragraph("0 words", table_text), Paragraph("Optimal empirical trade-off: high diversity with fluent story dialogue; LSTM and GRU comparable", table_text)],
        [Paragraph("<b>1.0</b>", table_text), Paragraph("0.855", table_text), Paragraph("0.992", table_text), Paragraph("0.0%", table_text), Paragraph("0 words", table_text), Paragraph("High lexical richness; slight syntactic drift across longer clauses", table_text)],
        [Paragraph("<b>1.5</b>", table_text), Paragraph("0.916", table_text), Paragraph("1.000", table_text), Paragraph("0.0%", table_text), Paragraph("0 words", table_text), Paragraph("High entropy; syntactic collapse into ungrammatical token sequences", table_text)],
    ]
    t_table = Table(t_data, colWidths=[35, 52, 52, 72, 55, 274])
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
        "• <b>LSTM vs. GRU & UNK Nuance:</b> LSTM leads across next-word prediction metrics (Val Loss 3.9903, PPL 54.07, Test Top-1 23.29%), with GRU performing closely behind (Val Loss 4.0353, Test Top-1 22.53%) with 11% fewer parameters. Test targets include 11.17% &lt;unk&gt; tokens due to train-only vocabulary fitting.<br/>"
        "• <b>Qualitative Consistency & Seed Origin:</b> Seed 1 comes from the training text, Seed 2 from held-out test text, and Seed 3 from validation text. Consistency was assessed qualitatively: at T=0.5, LSTM and GRU more frequently generate Carroll-style dialogue tags ('the march hare said'), though all models drift across multi-clause sequences without sustaining a persistent scene.<br/>"
        "• <b>Single-Seed Limitation & BiRNN:</b> SEED=42 ensures strict comparability; the 0.045 loss gap between LSTM and GRU falls within single-run variance margins. BiRNN's forward-backward state concatenation feeds complete context, but its 1,262,868 parameters overfit by epoch 2 (val loss 4.2804).<br/>"
        "• <b>Temperature Sampling:</b> T = 0.5 provides the optimal trade-off between lexical diversity (Distinct-1: 0.60–0.73) and phrase stability, achieving minimal trigram repetition (LSTM 5.0%, GRU 0.0%, BiRNN 1.0%, Simple RNN 6.1%).",
        body_style
    ))

    doc.build(story)
    print(f"Generated PDF document at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    docx_file = build_submission_docx()
    pdf_file = build_submission_pdf()
