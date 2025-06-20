#!/usr/bin/env python3
"""
Enhanced PDF Report Generator with Proper Math Rendering

Uses Pandoc to convert Markdown to PDF with beautiful LaTeX math equations.
Maintains the same professional styling as the original generator.
"""

import os
import sys
import subprocess
from pathlib import Path
import json

class MathPDFGenerator:
    def __init__(self, base_dir="/home/rexl1/riemann"):
        self.base_dir = Path(base_dir)
        self.analysis_dir = self.base_dir / "analysis"
        self.results_dir = self.base_dir / "results"
        self.markdown_file = self.results_dir / "universal_critical_restoration_conjecture_analysis.md"
        self.pdf_file = self.results_dir / "universal_critical_restoration_conjecture_analysis_with_math.pdf"
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
    def check_dependencies(self):
        """Check if required tools are available"""
        dependencies = ["pandoc"]
        missing = []
        
        for dep in dependencies:
            try:
                subprocess.run([dep, "--version"], capture_output=True, check=True)
                print(f"✓ {dep} is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing.append(dep)
                print(f"✗ {dep} is not available")
        
        if missing:
            print(f"Missing dependencies: {missing}")
            return False
        return True
    
    def create_pandoc_template(self):
        """Create a custom LaTeX template for professional formatting"""
        template_content = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{wrapfig}
\usepackage{float}
\usepackage{colortbl}
\usepackage{pdflscape}
\usepackage{tabu}
\usepackage{threeparttable}
\usepackage{threeparttablex}
\usepackage[normalem]{ulem}
\usepackage{makecell}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{hyperref}

% Page setup
\geometry{margin=2cm}
\pagestyle{fancy}
\fancyhf{}
\rhead{\thepage}
\renewcommand{\headrulewidth}{0pt}

% Colors
\definecolor{titlecolor}{HTML}{2c3e50}
\definecolor{sectioncolor}{HTML}{34495e}
\definecolor{subsectioncolor}{HTML}{7f8c8d}

% Title formatting
\usepackage{titlesec}
\titleformat{\section}
  {\normalfont\Large\bfseries\color{titlecolor}}
  {\thesection}{1em}{}
  [\color{titlecolor}\titlerule[2pt]]

\titleformat{\subsection}
  {\normalfont\large\bfseries\color{sectioncolor}}
  {\thesubsection}{1em}{}
  [\color{sectioncolor}\titlerule[1pt]]

\titleformat{\subsubsection}
  {\normalfont\normalsize\bfseries\color{subsectioncolor}}
  {\thesubsubsection}{1em}{}

% Math environments
\usepackage{mathtools}
\usepackage{physics}

% Better spacing
\setlength{\parskip}{6pt}
\setlength{\parindent}{0pt}

% Document
\begin{document}

$body$

\end{document}
"""
        template_file = self.analysis_dir / "report_template.tex"
        with open(template_file, 'w') as f:
            f.write(template_content)
        return template_file
    
    def convert_with_pandoc_webtex(self):
        """Convert using Pandoc with WebTeX for math rendering (no LaTeX required)"""
        try:
            print("Converting with Pandoc using WebTeX math rendering...")
            
            cmd = [
                "pandoc",
                str(self.markdown_file),
                "-o", str(self.pdf_file),
                "--pdf-engine=weasyprint",
                "--webtex",  # Use WebTeX for math rendering
                "--standalone",
                "--toc",
                "--toc-depth=3",
                "--variable", "geometry:margin=2cm",
                "--variable", "fontsize=11pt",
                "--variable", "papersize=a4",
                "--variable", "colorlinks=true",
                "--variable", "linkcolor=blue",
                "--variable", "urlcolor=blue",
                "--variable", "citecolor=blue"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ PDF with math generated: {self.pdf_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Pandoc with WebTeX failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def convert_with_pandoc_mathjax(self):
        """Convert using Pandoc with MathJax via HTML intermediate"""
        try:
            print("Converting with Pandoc using MathJax math rendering...")
            
            # First convert to HTML with MathJax
            html_file = self.results_dir / "temp_with_math.html"
            
            cmd_html = [
                "pandoc",
                str(self.markdown_file),
                "-o", str(html_file),
                "--mathjax",
                "--standalone",
                "--toc",
                "--toc-depth=3",
                "--css", "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css",
                "--metadata", "title=Universal Critical Restoration Conjecture Analysis"
            ]
            
            subprocess.run(cmd_html, capture_output=True, text=True, check=True)
            print(f"✓ HTML with MathJax created: {html_file}")
            
            # Then convert HTML to PDF using wkhtmltopdf or similar
            # For now, let's try a simpler approach
            return self.convert_html_to_pdf_with_chrome(html_file)
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Pandoc HTML conversion failed: {e}")
            return False
    
    def convert_html_to_pdf_with_chrome(self, html_file):
        """Convert HTML to PDF using Chrome/Chromium headless"""
        try:
            # Try different Chrome/Chromium commands
            chrome_commands = [
                "google-chrome",
                "chromium-browser", 
                "chromium",
                "chrome"
            ]
            
            chrome_cmd = None
            for cmd in chrome_commands:
                try:
                    subprocess.run([cmd, "--version"], capture_output=True, check=True)
                    chrome_cmd = cmd
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            if not chrome_cmd:
                print("✗ No Chrome/Chromium browser found for HTML to PDF conversion")
                return False
            
            print(f"Using {chrome_cmd} for PDF conversion...")
            
            cmd = [
                chrome_cmd,
                "--headless",
                "--disable-gpu",
                "--print-to-pdf=" + str(self.pdf_file),
                "--print-to-pdf-no-header",
                "--no-margins",
                "file://" + str(html_file.absolute())
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✓ PDF with rendered math created: {self.pdf_file}")
            
            # Clean up HTML file
            html_file.unlink()
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Chrome PDF conversion failed: {e}")
            return False
    
    def convert_with_katex(self):
        """Convert using Pandoc with KaTeX for math"""
        try:
            print("Converting with Pandoc using KaTeX math rendering...")
            
            cmd = [
                "pandoc",
                str(self.markdown_file),
                "-o", str(self.pdf_file),
                "--katex",
                "--standalone",
                "--toc",
                "--toc-depth=3",
                "--pdf-engine=weasyprint",
                "--variable", "geometry:margin=2cm",
                "--variable", "fontsize=11pt",
                "--variable", "papersize=a4"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ PDF with KaTeX math generated: {self.pdf_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Pandoc with KaTeX failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def generate_pdf(self):
        """Try multiple methods to generate PDF with proper math rendering"""
        if not self.check_dependencies():
            print("✗ Missing required dependencies")
            return False
        
        if not self.markdown_file.exists():
            print(f"✗ Markdown file not found: {self.markdown_file}")
            return False
        
        print(f"📄 Source: {self.markdown_file}")
        print(f"📊 Target: {self.pdf_file}")
        
        # Try methods in order of preference
        methods = [
            ("MathJax via HTML", self.convert_with_pandoc_mathjax),
            ("WebTeX", self.convert_with_pandoc_webtex),
            ("KaTeX", self.convert_with_katex)
        ]
        
        for method_name, method_func in methods:
            print(f"\n🔄 Trying {method_name}...")
            if method_func():
                print(f"✅ Successfully generated PDF using {method_name}!")
                
                # Check file size
                if self.pdf_file.exists():
                    size_mb = self.pdf_file.stat().st_size / 1024 / 1024
                    print(f"📊 PDF size: {size_mb:.2f} MB")
                
                return True
            else:
                print(f"❌ {method_name} failed, trying next method...")
        
        print("❌ All methods failed to generate PDF with math")
        return False

def main():
    print("=== PDF Generator with Math Rendering ===")
    
    generator = MathPDFGenerator()
    success = generator.generate_pdf()
    
    if success:
        print("\n🎉 PDF with beautiful math equations generated successfully!")
        print(f"📁 Location: {generator.pdf_file}")
    else:
        print("\n😞 Failed to generate PDF with proper math rendering")
        print("Try installing additional dependencies or check the markdown file")

if __name__ == "__main__":
    main()
