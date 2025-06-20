#!/usr/bin/env python3
"""
PDF Report Generator for Universal Critical Restoration Conjecture Analysis

Converts the Markdown report to a professional PDF with embedded images.
Uses markdown2 + weasyprint for high-quality PDF generation.
"""

import os
import sys
import subprocess
from pathlib import Path
import markdown
import pdfkit
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

class PDFReportGenerator:
    def __init__(self, base_dir="/home/rexl1/riemann"):
        self.base_dir = Path(base_dir)
        self.analysis_dir = self.base_dir / "analysis"
        self.results_dir = self.base_dir / "results"
        self.markdown_file = self.results_dir / "universal_critical_restoration_conjecture_analysis.md"
        self.pdf_file = self.results_dir / "universal_critical_restoration_conjecture_analysis.pdf"
        self.html_file = self.analysis_dir / "temp_report.html"
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
    def install_dependencies(self):
        """Install required Python packages using conda"""
        conda_packages = [
            "markdown",
            "weasyprint", 
            "pymdown-extensions"
        ]
        
        pip_packages = [
            "pdfkit",
            "python-markdown-math"
        ]
        
        print("Installing PDF generation dependencies with conda...")
        
        # Try conda first for better compatibility
        for package in conda_packages:
            try:
                subprocess.run(["conda", "install", "-y", "-c", "conda-forge", package], 
                             check=True, capture_output=True)
                print(f"✓ {package} installed via conda")
            except subprocess.CalledProcessError as e:
                print(f"⚠ Failed to install {package} via conda: {e}")
                # Fallback to pip
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                 check=True, capture_output=True)
                    print(f"✓ {package} installed via pip (fallback)")
                except subprocess.CalledProcessError as pip_e:
                    print(f"⚠ Failed to install {package} via pip: {pip_e}")
        
        # Use pip for packages not available in conda
        print("Installing remaining packages with pip...")
        for package in pip_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                print(f"✓ {package} installed via pip")
            except subprocess.CalledProcessError as e:
                print(f"⚠ Failed to install {package} via pip: {e}")
                if package == "python-markdown-math":
                    print("  Note: Math rendering may be limited without this package")
    
    def convert_to_html(self):
        """Convert Markdown to HTML with math support"""
        print("Converting Markdown to HTML...")
        
        # Read markdown content
        if not self.markdown_file.exists():
            raise FileNotFoundError(f"Markdown file not found: {self.markdown_file}")
            
        content = self.markdown_file.read_text(encoding='utf-8')
        
        # Configure markdown with extensions
        md = markdown.Markdown(extensions=[
            'extra',          # Tables, code blocks, etc.
            'codehilite',     # Syntax highlighting
            'toc',            # Table of contents
            'pymdownx.arithmatex',  # Math support
            'pymdownx.superfences',  # Advanced code blocks
        ], extension_configs={
            'pymdownx.arithmatex': {
                'generic': True
            }
        })
        
        # Convert to HTML
        html_body = md.convert(content)
        
        # Create complete HTML document with CSS styling
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Universal Critical Restoration Conjecture Analysis</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
            @bottom-center {{
                content: counter(page);
            }}
        }}
        
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            color: #333;
            max-width: none;
            margin: 0;
            padding: 0;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            page-break-before: always;
        }}
        
        h1:first-child {{
            page-break-before: avoid;
        }}
        
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        
        h3 {{
            color: #7f8c8d;
            margin-top: 25px;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        
        table, th, td {{
            border: 1px solid #ddd;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
        }}
        
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #ecf0f1;
            font-style: italic;
        }}
        
        .math {{
            text-align: center;
            margin: 20px 0;
        }}
        
        /* Page breaks */
        .page-break {{
            page-break-before: always;
        }}
        
        /* Print optimizations */
        @media print {{
            body {{
                font-size: 11pt;
            }}
            
            h1 {{
                font-size: 18pt;
            }}
            
            h2 {{
                font-size: 14pt;
            }}
            
            h3 {{
                font-size: 12pt;
            }}
        }}
    </style>
    
    <!-- MathJax for mathematical expressions -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            }}
        }};
    </script>
</head>
<body>
{html_body}
</body>
</html>
"""
        
        # Write HTML file
        self.html_file.write_text(html_content, encoding='utf-8')
        print(f"✓ HTML generated: {self.html_file}")
        
    def convert_to_pdf_weasyprint(self):
        """Convert HTML to PDF using WeasyPrint (best quality)"""
        try:
            print("Converting HTML to PDF using WeasyPrint...")
            
            # Configure fonts
            font_config = FontConfiguration()
            
            # Custom CSS for better PDF formatting
            css = CSS(string='''
                @page {
                    size: A4;
                    margin: 2cm;
                }
                img {
                    max-width: 100%;
                    height: auto;
                }
            ''', font_config=font_config)
            
            # Convert to PDF
            HTML(filename=str(self.html_file)).write_pdf(
                str(self.pdf_file),
                stylesheets=[css],
                font_config=font_config
            )
            
            print(f"✓ PDF generated: {self.pdf_file}")
            return True
            
        except Exception as e:
            print(f"⚠ WeasyPrint failed: {e}")
            return False
    
    def convert_to_pdf_pdfkit(self):
        """Convert HTML to PDF using pdfkit (fallback)"""
        try:
            print("Converting HTML to PDF using pdfkit...")
            
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None,
                'enable-local-file-access': None
            }
            
            pdfkit.from_file(str(self.html_file), str(self.pdf_file), options=options)
            print(f"✓ PDF generated: {self.pdf_file}")
            return True
            
        except Exception as e:
            print(f"⚠ pdfkit failed: {e}")
            return False
    
    def cleanup(self):
        """Remove temporary files"""
        if self.html_file.exists():
            self.html_file.unlink()
            print("✓ Temporary HTML file cleaned up")
    
    def generate_pdf(self):
        """Generate PDF report with all images embedded"""
        print("=== Universal Critical Restoration Conjecture PDF Generator ===")
        print(f"Input: {self.markdown_file}")
        print(f"Output: {self.pdf_file}")
        
        try:
            # Step 1: Install dependencies
            self.install_dependencies()
            
            # Step 2: Convert to HTML
            self.convert_to_html()
            
            # Step 3: Convert to PDF (try WeasyPrint first, then pdfkit)
            success = self.convert_to_pdf_weasyprint()
            if not success:
                success = self.convert_to_pdf_pdfkit()
            
            if success:
                print(f"\n🎉 PDF Report Generated Successfully!")
                print(f"📄 Location: {self.pdf_file}")
                print(f"📊 Size: {self.pdf_file.stat().st_size / 1024 / 1024:.2f} MB")
            else:
                print("\n❌ PDF generation failed with both methods")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            
        finally:
            # Clean up temporary files
            self.cleanup()

def main():
    """Main function"""
    generator = PDFReportGenerator()
    generator.generate_pdf()

if __name__ == "__main__":
    main()
