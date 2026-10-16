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
        "experimental conditions on Lewis Carroll's 'Alice's Adventures in Wonderland' corpus:\n"
        "1. Simple (Vanilla) RNN\n"
        "2. Bidirectional RNN\n"
        "3. Long Short-Term Memory (LSTM)\n"
        "4. Gated Recurrent Unit (GRU)\n\n"
        "Each architecture was trained to predict the succeeding word given an 8-token sliding context window. "
        "The models are quantitatively evaluated on cross-entropy loss, perplexity, top-1 accuracy, top-5 candidate accuracy, and training wall-clock time, "
        "followed by qualitative evaluation of text generation across four temperature regimes (T = 0.2, 0.5, 1.0, 1.5)."
    )
    
    # Section: Quantitative Results
    h2 = doc.add_heading(level=1)
    r_h2 = h2.add_run("2. Quantitative Evaluation Summary")
    r_h2.font.size = Pt(13)
    r_h2.font.bold = True
    r_h2.font.color.rgb = RGBColor(30, 60, 114)
    
    q_table = doc.add_table(rows=5, cols=8)
    q_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    q_table.autofit = True
    
    q_headers = ["Architecture", "Parameters", "Val Loss", "Perplexity", "Top-1 Acc", "Top-5 Acc", "Total Time", "Avg Epoch Time"]
    for j, h in enumerate(q_headers):
        cell = q_table.rows[0].cells[j]
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.font.bold = True
        run.font.size = Pt(9.0)
        
    q_rows = [
        ("Simple RNN", "724,097", "4.9572", "142.20", "20.28%", "40.72%", "54.1s", "4.51s"),
        ("Bidirectional RNN", "1,101,313", "5.1979", "180.89", "19.37%", "40.10%", "59.1s", "4.92s"),
        ("LSTM", "823,169", "4.8746", "130.92", "20.22%", "42.02%", "48.1s", "4.00s"),
        ("GRU", "790,145", "4.9834", "145.98", "20.09%", "40.81%", "50.4s", "4.19s"),
    ]
    
    for i, row_data in enumerate(q_rows):
        for j, val in enumerate(row_data):
            cell = q_table.rows[i+1].cells[j]
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.size = Pt(9.0)
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
        "• Low Temperature (T = 0.2): Highly confident predictions. Yields grammatically consistent prose, but exhibits cyclical repetition ('and she had to herself, and she had to her').\n"
        "• Moderate Temperature (T = 0.5): Optimal trade-off between coherence and diversity (Distinct-1: ~0.75, Distinct-2: ~0.98). Captures authentic dialogue exchanges between characters like the Hatter, Queen, and Alice.\n"
        "• Neutral Temperature (T = 1.0): High vocabulary diversity with occasional grammatical divergence across multi-clause boundaries.\n"
        "• High Temperature (T = 1.5): Excessive entropy leading to syntactic breakdown and random token sequences.\n\n"
        "Conclusion: The LSTM model achieved the lowest validation loss (4.8746), lowest perplexity (130.92), and highest Top-5 candidate accuracy (42.02%) with the lowest execution latency per epoch (4.00s). Combined with T = 0.5 sampling, LSTM provides the most effective autoregressive text generation for this dataset."
    )
    
    output_path = os.path.join(os.getcwd(), output_filename)
    doc.save(output_path)
    print(f"Generated Word document at: {output_path}")
    return output_path

if __name__ == "__main__":
    build_submission_doc()
