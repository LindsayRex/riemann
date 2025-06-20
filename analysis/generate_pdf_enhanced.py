#!/usr/bin/env python3
"""
Enhanced PDF Report Generator with Proper Math Rendering

Keeps the excellent formatting from the original generator but adds proper math rendering
by preprocessing LaTeX equations before conversion.
"""

import os
import sys
import subprocess
import re
import tempfile
from pathlib import Path
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

class MathRenderer:
    """Handles LaTeX math rendering using available tools"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def render_latex_to_svg(self, latex_expr, display_mode=False):
        """Convert LaTeX expression to SVG using available tools"""
        try:
            # Try using Pandoc with MathML output first
            pandoc_input = f"${latex_expr}$" if not display_mode else f"$${latex_expr}$$"
            
            result = subprocess.run([
                'pandoc', 
                '--from=markdown+tex_math_dollars',
                '--to=html',
                '--mathml'
            ], input=pandoc_input, text=True, capture_output=True)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                # Fallback: return formatted text
                return f'<span class="math-fallback">{"$$" if display_mode else "$"}{latex_expr}{"$$" if display_mode else "$"}</span>'
                
        except Exception as e:
            print(f"Math rendering error: {e}")
            return f'<span class="math-fallback">{"$$" if display_mode else "$"}{latex_expr}{"$$" if display_mode else "$"}</span>'

class EnhancedPDFGenerator:
    def __init__(self, base_dir="/home/rexl1/riemann"):
        self.base_dir = Path(base_dir)
        self.analysis_dir = self.base_dir / "analysis"
        self.results_dir = self.base_dir / "results"
        self.markdown_file = self.results_dir / "universal_critical_restoration_conjecture_analysis.md"
        self.pdf_file = self.results_dir / "universal_critical_restoration_conjecture_analysis_enhanced.pdf"
        self.math_renderer = MathRenderer()
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
    def preprocess_math(self, content):
        """Preprocess LaTeX math expressions in markdown content"""
        
        # Pattern for display math ($$...$$)
        def replace_display_math(match):
            latex = match.group(1)
            return self.math_renderer.render_latex_to_svg(latex, display_mode=True)
        
        # Pattern for inline math ($...$)
        def replace_inline_math(match):
            latex = match.group(1)
            return self.math_renderer.render_latex_to_svg(latex, display_mode=False)
        
        # Replace display math first ($$...$$)
        content = re.sub(r'\$\$(.*?)\$\$', replace_display_math, content, flags=re.DOTALL)
        
        # Replace inline math ($...$) - be careful not to match $$ that's already processed
        content = re.sub(r'(?<!\$)\$([^$\n]+?)\$(?!\$)', replace_inline_math, content)
        
        return content
        
    def generate_html(self):
        """Convert markdown to HTML with proper math rendering"""
        if not self.markdown_file.exists():
            raise FileNotFoundError(f"Markdown file not found: {self.markdown_file}")
            
        print("Reading markdown content...")
        content = self.markdown_file.read_text(encoding='utf-8')
        
        print("Preprocessing math expressions...")
        content = self.preprocess_math(content)
        
        # Configure markdown with extensions
        md = markdown.Markdown(extensions=[
            'extra',          # Tables, code blocks, etc.
            'codehilite',     # Syntax highlighting
            'toc',            # Table of contents
            'pymdownx.superfences',  # Advanced code blocks
        ])
        
        # Convert to HTML
        html_body = md.convert(content)
        
        # Create complete HTML document with enhanced CSS styling
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
        
        /* Math styling - enhanced for better visibility */
        .math {{
            text-align: center;
            margin: 20px 0;
            font-family: 'Times New Roman', serif;
        }}
        
        .math-fallback {{
            font-family: 'Times New Roman', serif;
            font-style: italic;
            background-color: #f9f9f9;
            padding: 2px 4px;
            border-radius: 3px;
            border: 1px solid #e0e0e0;
        }}
        
        /* MathML styling */
        math {{
            font-family: 'Times New Roman', serif;
            font-size: 1em;
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
</head>
<body>
{html_body}
</body>
</html>"""
        
        return html_content
        
    def convert_to_pdf(self):
        """Generate PDF from HTML using WeasyPrint"""
        
        try:
            print("Generating HTML with math rendering...")
            html_content = self.generate_html()
            
            print("Converting to PDF with WeasyPrint...")
            
            # Configure font handling
            font_config = FontConfiguration()
            
            # Convert HTML to PDF
            html_doc = HTML(string=html_content, base_url=str(self.analysis_dir))
            
            html_doc.write_pdf(
                str(self.pdf_file),
                font_config=font_config,
                presentational_hints=True
            )
            
            print(f"✅ PDF generated successfully: {self.pdf_file}")
            
            # Check file size
            size_mb = self.pdf_file.stat().st_size / 1024 / 1024
            print(f"📄 PDF size: {size_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            return False

def main():
    """Main function to generate PDF report"""
    
    print("🔧 Enhanced PDF Generator with Math Rendering")
    print("=" * 50)
    
    generator = EnhancedPDFGenerator()
    
    if generator.convert_to_pdf():
        print("\n🎉 PDF report generated successfully!")
        print(f"📍 Location: {generator.pdf_file}")
    else:
        print("\n❌ Failed to generate PDF report")
        sys.exit(1)

if __name__ == "__main__":
    main()
