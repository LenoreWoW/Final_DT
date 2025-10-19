#!/usr/bin/env python3
"""
LaTeX Compilation Script for Grant Report with Bibliography

This script compiles the INTERIM_GRANT_REPORT.tex with proper bibliography
integration following LaTeX best practices.
"""

import subprocess
import os
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle output"""
    print(f"📄 {description}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd='/Users/hassanalsahli/Desktop/Final_DT'
        )
        if result.returncode != 0:
            print(f"   ❌ Error: {result.stderr}")
            return False
        else:
            print(f"   ✅ Success")
            return True
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def compile_latex_with_bibliography():
    """Compile LaTeX document with bibliography following best practices"""
    print("🎓 COMPILING GRANT REPORT WITH BIBLIOGRAPHY")
    print("=" * 50)
    
    # Check if files exist
    files_to_check = [
        "INTERIM_GRANT_REPORT.tex",
        "academic_bibliography.bib"
    ]
    
    for file in files_to_check:
        if not Path(file).exists():
            print(f"❌ Missing file: {file}")
            return False
    
    print("✅ All required files found")
    print()
    
    # LaTeX compilation sequence for bibliography
    compilation_steps = [
        ("pdflatex INTERIM_GRANT_REPORT.tex", "First LaTeX compilation"),
        ("bibtex INTERIM_GRANT_REPORT", "Bibliography processing"),
        ("pdflatex INTERIM_GRANT_REPORT.tex", "Second LaTeX compilation"),
        ("pdflatex INTERIM_GRANT_REPORT.tex", "Final LaTeX compilation")
    ]
    
    success_count = 0
    for command, description in compilation_steps:
        if run_command(command, description):
            success_count += 1
        else:
            print(f"\n⚠️ Compilation step failed: {description}")
            print("This is expected if LaTeX is not installed.")
            break
    
    print(f"\n📊 Compilation Results: {success_count}/{len(compilation_steps)} steps completed")
    
    # Check if PDF was generated
    if Path("INTERIM_GRANT_REPORT.pdf").exists():
        pdf_size = Path("INTERIM_GRANT_REPORT.pdf").stat().st_size
        print(f"✅ PDF generated successfully: {pdf_size:,} bytes")
        return True
    else:
        print("📋 PDF not generated (LaTeX may not be installed)")
        print("   Use Overleaf for compilation:")
        print("   1. Upload INTERIM_GRANT_REPORT.tex and academic_bibliography.bib")
        print("   2. Overleaf will automatically compile with bibliography")
        return False

def validate_bibliography():
    """Validate bibliography entries match citations"""
    print("\n🔍 VALIDATING BIBLIOGRAPHY")
    print("=" * 30)
    
    # Extract citations from LaTeX file
    try:
        with open("INTERIM_GRANT_REPORT.tex", 'r') as f:
            tex_content = f.read()
        
        import re
        citations = re.findall(r'\\cite\{([^}]+)\}', tex_content)
        
        # Flatten citation lists (some citations have multiple keys)
        all_citations = []
        for citation in citations:
            all_citations.extend([c.strip() for c in citation.split(',')])
        
        unique_citations = sorted(set(all_citations))
        print(f"📋 Found {len(unique_citations)} unique citations:")
        for citation in unique_citations:
            print(f"   • {citation}")
        
        # Check bibliography file
        with open("academic_bibliography.bib", 'r') as f:
            bib_content = f.read()
        
        bib_entries = re.findall(r'@\w+\{([^,]+),', bib_content)
        print(f"\n📚 Found {len(bib_entries)} bibliography entries:")
        for entry in sorted(bib_entries):
            print(f"   • {entry}")
        
        # Check for missing entries
        missing_entries = set(unique_citations) - set(bib_entries)
        if missing_entries:
            print(f"\n❌ Missing bibliography entries:")
            for missing in missing_entries:
                print(f"   • {missing}")
            return False
        else:
            print(f"\n✅ All citations have corresponding bibliography entries")
            return True
    
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def generate_compilation_guide():
    """Generate compilation guide for different scenarios"""
    guide = """
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
- Check that all \\cite{} entries exist in academic_bibliography.bib
- Run complete sequence: pdflatex → bibtex → pdflatex → pdflatex

**LaTeX not installed?**
- Use Overleaf (easiest option)
- Install MacTeX/TeXLive for local compilation

## Files Required
- INTERIM_GRANT_REPORT.tex (main document)
- academic_bibliography.bib (references database)
"""
    
    with open("COMPILATION_GUIDE.md", 'w') as f:
        f.write(guide)
    
    print("📋 Created COMPILATION_GUIDE.md with detailed instructions")

if __name__ == "__main__":
    # Validate bibliography first
    bib_valid = validate_bibliography()
    
    # Attempt compilation
    if bib_valid:
        compile_success = compile_latex_with_bibliography()
    else:
        print("⚠️ Bibliography validation failed - fixing required before compilation")
        compile_success = False
    
    # Generate guide regardless
    generate_compilation_guide()
    
    print(f"\n🏆 FINAL STATUS:")
    print(f"   Bibliography: {'✅ Valid' if bib_valid else '❌ Needs fixes'}")
    print(f"   Compilation: {'✅ Success' if compile_success else '📋 Use Overleaf'}")
    print(f"   Guide: ✅ Created COMPILATION_GUIDE.md")
    
    if not compile_success:
        print(f"\n💡 RECOMMENDED NEXT STEP:")
        print(f"   Upload files to Overleaf for automatic compilation with bibliography")
