import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def build_submission_doc(
    student_name="Saksham Gupta",
    reg_num="[Your Registration Number Here]",
    repo_url="https://github.com/sakshamgupta/autoregressive-text-generation-fgai",
    output_filename="SakshamGupta_RegNum_FGAI_Assignment2.docx"
):
    doc = Document()
    
    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Document Header Title
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
        ("System Title", "Autoregressive Next-Word Prediction & Multi-Temperature Generation System"),
        ("Dataset Name & Source", "Tiny Shakespeare Corpus\nhttps://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"),
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
        r0.font.size = Pt(10.5)
        
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.size = Pt(10.5)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(18)
    
    # Section: Executive Summary & System Description
    h1 = doc.add_heading(level=1)
    r_h1 = h1.add_run("1. System Overview & Problem Statement")
    r_h1.font.size = Pt(13)
    r_h1.font.bold = True
    r_h1.font.color.rgb = RGBColor(30, 60, 114)
    
    doc.add_paragraph(
        "This project implements and evaluates an autoregressive next-word language generation system. "
        "Four distinct recurrent neural network architectures were implemented, trained, and tested under identical "
        "experimental conditions on the Tiny Shakespeare corpus:\n"
        "1. Simple (Vanilla) RNN\n"
        "2. Bidirectional RNN\n"
        "3. Long Short-Term Memory (LSTM)\n"
        "4. Gated Recurrent Unit (GRU)\n\n"
        "Each architecture was trained to predict the succeeding word given an 8-token context window. "
        "The models are quantitatively evaluated on cross-entropy loss, perplexity, top-1 accuracy, and training wall-clock time, "
        "followed by qualitative evaluation of text generation across four temperature regimes (T = 0.2, 0.5, 1.0, 1.5)."
    )
    
    # Section: Quantitative Results
    h2 = doc.add_heading(level=1)
    r_h2 = h2.add_run("2. Quantitative Evaluation Summary")
    r_h2.font.size = Pt(13)
    r_h2.font.bold = True
    r_h2.font.color.rgb = RGBColor(30, 60, 114)
    
    q_table = doc.add_table(rows=5, cols=7)
    q_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    q_table.autofit = True
    
    q_headers = ["Architecture", "Parameters", "Val Loss", "Perplexity", "Val Accuracy", "Total Time", "Avg Epoch Time"]
    for j, h in enumerate(q_headers):
        cell = q_table.rows[0].cells[j]
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.font.bold = True
        run.font.size = Pt(9.5)
        
    q_rows = [
        ("Simple RNN", "1,318,024", "5.1671", "175.40", "16.34%", "37.9s", "3.79s"),
        ("Bidirectional RNN", "1,991,048", "5.3461", "209.79", "15.49%", "42.7s", "4.27s"),
        ("LSTM", "1,417,096", "5.0745", "159.89", "16.94%", "36.0s", "3.58s"),
        ("GRU", "1,384,072", "5.1609", "174.32", "16.39%", "37.3s", "3.72s"),
    ]
    
    for i, row_data in enumerate(q_rows):
        for j, val in enumerate(row_data):
            cell = q_table.rows[i+1].cells[j]
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.size = Pt(9.5)
            if j == 0:
                run.font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Section: Key Findings & Temperature Analysis
    h3 = doc.add_heading(level=1)
    r_h3 = h3.add_run("3. Sampling Temperature & Qualitative Findings")
    r_h3.font.size = Pt(13)
    r_h3.font.bold = True
    r_h3.font.color.rgb = RGBColor(30, 60, 114)
    
    doc.add_paragraph(
        "• Low Temperature (T = 0.2): Emphasizes highest-probability tokens. Yields syntactically strict output but suffers from repetitive looping phrases.\n"
        "• Moderate Temperature (T = 0.5): Best balance of coherence and diversity (Distinct-1: ~0.71, Distinct-2: ~0.95). Captures Shakespearean theatrical turns of phrase.\n"
        "• Neutral Temperature (T = 1.0): High vocabulary diversity with occasional grammatical divergence.\n"
        "• High Temperature (T = 1.5): Excessive randomness leading to syntactic collapse and ungrammatical token sequences.\n\n"
        "Conclusion: The LSTM model achieved the lowest validation loss (5.0745), lowest perplexity (159.89), and highest next-word accuracy (16.94%) while maintaining the lowest training latency per epoch (3.58s). In combination with T = 0.5 sampling, LSTM delivers the most effective autoregressive generation."
    )
    
    output_path = os.path.join(os.getcwd(), output_filename)
    doc.save(output_path)
    print(f"Generated Word document at: {output_path}")
    return output_path

if __name__ == "__main__":
    build_submission_doc()
