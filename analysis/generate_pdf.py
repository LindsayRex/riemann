#!/usr/bin/env python3
"""
PDF Generator with Math Rendering

Combines Pandoc + Custom HTML Template + MathJax + Chrome headless
for beautiful styling AND proper math rendering.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

class PDFGeneratorWithMath:
    def __init__(self, base_dir="/home/rexl1/riemann"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.analysis_dir = self.base_dir / "analysis"
        self.markdown_file = self.results_dir / "universal_critical_restoration_conjecture_analysis.md"
        self.pdf_file = self.results_dir / "universal_critical_restoration_conjecture_analysis.pdf"
        
    def create_html_template(self):
        """Create custom HTML template with your beautiful styling + MathJax"""
        
        template_content = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>$title$</title>
    
    <!-- MathJax -->
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    
    <style>
        body {
            font-family: Times, serif;
            line-height: 1.6;
            color: #333;
            margin: 20px;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 5px;
        }
        
        h3 {
            color: #7f8c8d;
        }
        
        img {
            max-width: 100%;
            margin: 20px auto;
            display: block;
            border: 1px solid #ddd;
            padding: 5px;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
        }
        
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        pre {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            padding: 15px;
            overflow-x: auto;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #ecf0f1;
        }
    </style>
</head>
<body>
$body$
</body>
</html>'''
        
        template_file = self.analysis_dir / "custom_template.html"
        template_file.write_text(template_content)
        return template_file
    
    def convert_to_html(self, template_file):
        """Convert markdown to HTML with MathJax using custom template"""
        
        html_file = self.results_dir / "temp_report_with_math.html"
        
        cmd = [
            "pandoc",
            str(self.markdown_file),
            "-o", str(html_file),
            "--from", "markdown+tex_math_dollars",
            "--to", "html5",
            "--template", str(template_file),
            "--mathjax",
            "--standalone",
            "--toc",
            "--toc-depth=3",
            "--metadata", "title=Universal Critical Restoration Conjecture Analysis"
        ]
        
        print("🔄 Converting markdown to HTML with MathJax...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.results_dir)
        
        if result.returncode != 0:
            print(f"❌ Pandoc HTML conversion failed: {result.stderr}")
            return None
        
        print(f"✅ HTML with math created: {html_file}")
        return html_file
    
    def convert_html_to_pdf(self, html_file):
        """Convert HTML to PDF using Chrome headless"""
        
        # Try different Chrome commands
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
                print(f"✅ Found {cmd}")
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        if not chrome_cmd:
            print("❌ No Chrome/Chromium browser found")
            return False
        
        print("🔄 Converting HTML to PDF with Chrome...")
        
        cmd = [
            chrome_cmd,
            "--headless",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--run-all-compositor-stages-before-draw",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features=TranslateUI",
            "--disable-extensions",
            "--disable-ipc-flooding-protection",
            "--print-to-pdf=" + str(self.pdf_file),
            "--print-to-pdf-no-header",
            "--virtual-time-budget=10000",  # Wait for MathJax
            "file://" + str(html_file.absolute())
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and self.pdf_file.exists():
            print(f"✅ PDF created: {self.pdf_file}")
            
            # Clean up temporary HTML
            html_file.unlink()
            
            return True
        else:
            print(f"❌ Chrome PDF conversion failed: {result.stderr}")
            return False
    
    def generate_pdf(self):
        """Generate perfect PDF with beautiful styling AND math"""
        
        print("🎯 PDF Generator with Math Rendering")
        print("=" * 50)
        
        if not self.markdown_file.exists():
            print(f"❌ Markdown file not found: {self.markdown_file}")
            return False
        
        try:
            # Step 1: Create custom HTML template
            print("📝 Creating custom HTML template...")
            template_file = self.create_html_template()
            
            # Step 2: Convert markdown to HTML with MathJax
            html_file = self.convert_to_html(template_file)
            if not html_file:
                return False
            
            # Step 3: Convert HTML to PDF with Chrome
            if not self.convert_html_to_pdf(html_file):
                return False
            
            # Check final result
            if self.pdf_file.exists():
                size_mb = self.pdf_file.stat().st_size / 1024 / 1024
                print(f"📊 PDF size: {size_mb:.2f} MB")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    generator = PDFGeneratorWithMath()
    
    success = generator.generate_pdf()
    
    if success:
        print("\n🎉 SUCCESS! PDF with math generated!")
        print(f"📁 Location: {generator.pdf_file}")
        print("\n✅ What you got:")
        print("• 🎨 Beautiful CSS styling")
        print("• 🧮 Perfect LaTeX math rendering via MathJax")
        print("• 🖼️  All images included")
        print("• 📋 Table of contents")
        print("• 📄 Professional formatting")
    else:
        print("\n❌ PDF generation failed")

if __name__ == "__main__":
    main()
