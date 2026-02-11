
# 📄 GRANT REPORT COMPILATION GUIDE

## Method 1: Overleaf (Recommended - No Installation Required)

1. **Go to overleaf.com**
2. **Create new project**
3. **Upload files:**
   - INTERIM_GRANT_REPORT.tex
   - academic_bibliography.bib
4. **Click "Recompile"**
5. **Download PDF**

✅ Overleaf automatically handles bibliography compilation

## Method 2: Local LaTeX Installation

### macOS:
```bash
# Install MacTeX
brew install --cask mactex

# Compile with bibliography
pdflatex INTERIM_GRANT_REPORT.tex
bibtex INTERIM_GRANT_REPORT
pdflatex INTERIM_GRANT_REPORT.tex
pdflatex INTERIM_GRANT_REPORT.tex
```

### Linux/Ubuntu:
```bash
# Install TeXLive
sudo apt-get install texlive-full

# Compile with bibliography
pdflatex INTERIM_GRANT_REPORT.tex
bibtex INTERIM_GRANT_REPORT
pdflatex INTERIM_GRANT_REPORT.tex
pdflatex INTERIM_GRANT_REPORT.tex
```

## Troubleshooting

**Missing references?**
- Ensure bibtex step was successful
- Check that all \cite{} entries exist in academic_bibliography.bib
- Run complete sequence: pdflatex → bibtex → pdflatex → pdflatex

**LaTeX not installed?**
- Use Overleaf (easiest option)
- Install MacTeX/TeXLive for local compilation

## Files Required
- INTERIM_GRANT_REPORT.tex (main document)
- academic_bibliography.bib (references database)
