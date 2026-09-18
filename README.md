# Pharmaceutical Packaging Artwork Comparison Application

Offline desktop tool for comparing "original" vs "suspect" pharmaceutical packaging
artwork (PDF/image) at pixel, text, graphics, and barcode/Braille level. See
`SRS_Artwork_Comparison_App .docx` for requirements and `Architecture_Document.md`
for the technical design this codebase follows.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
```

**Tesseract OCR** (used by the text/OCR detector, Phase 4) is a separate native
binary — install it yourself and make sure `tesseract` is on `PATH`, or point
`pytesseract.pytesseract.tesseract_cmd` at it. Not required until Phase 4.

## Run

```bash
python -m app.main
```

## Test

```bash
pytest
```

`tests/golden_pairs/` is where hand-verified original/suspect artwork fixture
pairs go for the regression suite (Architecture §9) — empty until real artwork
samples are added.
