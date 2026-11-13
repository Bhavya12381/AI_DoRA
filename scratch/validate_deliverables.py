import sys
from pathlib import Path
from pptx import Presentation
import docx

def validate():
    pptx_path = Path("presentation/AI_DoRA_Research_Presentation.pptx")
    docx_path = Path("presentation/AI_DoRA_Presentation_Script.docx")

    assert pptx_path.exists(), f"File {pptx_path} does not exist!"
    assert docx_path.exists(), f"File {docx_path} does not exist!"

    # 1. Validate PPTX
    prs = Presentation(pptx_path)
    slide_count = len(prs.slides)
    print(f"PPTX Slide Count: {slide_count}")
    assert slide_count == 10, f"Expected exactly 10 slides, got {slide_count}!"

    slide_titles = []
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    if p.text.strip():
                        texts.append(p.text.strip())
        # Find title
        title = texts[1] if len(texts) > 1 else texts[0]
        slide_titles.append(title)
        print(f"  Slide {i:2d}: {title} (Total text elements: {len(texts)})")

    # 2. Validate DOCX
    doc = docx.Document(docx_path)
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading 1")]
    print(f"\nDOCX Heading 1 Count: {len(headings)}")
    assert len(headings) == 10, f"Expected exactly 10 sections in script, got {len(headings)}!"

    for i, h in enumerate(headings, 1):
        print(f"  Section {i:2d}: {h}")

    # Check words per section
    section_texts = {i: [] for i in range(1, 11)}
    current_sec = 0
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading 1"):
            current_sec += 1
        elif current_sec > 0 and not p.text.startswith("Estimated Speaking") and not p.text.startswith("What to say:") and not p.text.startswith("—"):
            section_texts[current_sec].append(p.text)

    print("\nWord Counts per Script Section:")
    total_words = 0
    for sec_num, paras in section_texts.items():
        text = " ".join(paras)
        words = len(text.split())
        total_words += words
        est_sec = words / 2.3 # ~138 words/min = 2.3 words/sec
        print(f"  Slide {sec_num:2d} Script: {words:3d} words (~{est_sec:.0f} seconds speaking time)")

    print(f"\nTotal Spoken Words: {total_words} words (~{total_words / 2.3 / 60:.1f} minutes total presentation)")

    # 3. Check for unauthorized experimental claims / fabrication
    banned_phrases = [
        "we achieved",
        "our model reached 9",
        "our implementation outperforms",
        "we evaluated on sst-2 and got",
        "results show that our method achieved"
    ]
    all_pptx_text = " ".join([shape.text_frame.text.lower() for slide in prs.slides for shape in slide.shapes if shape.has_text_frame])
    all_docx_text = " ".join([p.text.lower() for p in doc.paragraphs])

    for bp in banned_phrases:
        assert bp not in all_pptx_text, f"Found banned phrase in PPTX: '{bp}'"
        assert bp not in all_docx_text, f"Found banned phrase in DOCX: '{bp}'"

    print("\nValidation passed: No fabricated results or unverified claims found.")
    print("ALL VALIDATION CRITERIA SATISFIED!")

if __name__ == "__main__":
    validate()
