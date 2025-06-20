#!/usr/bin/env python3
"""
Verification script to confirm both markdown and PDF generation work with results folder
"""

from pathlib import Path

def main():
    print("=== RESULTS FOLDER VERIFICATION ===")
    
    base_dir = Path("/home/rexl1/riemann")
    results_dir = base_dir / "results"
    analysis_dir = base_dir / "analysis"
    
    # Check files
    markdown_file = results_dir / "universal_critical_restoration_conjecture_analysis.md"
    pdf_file = results_dir / "universal_critical_restoration_conjecture_analysis.pdf"
    images_dir = analysis_dir / "images"
    
    print(f"📁 Results Directory: {results_dir}")
    print(f"📄 Markdown: {'✅' if markdown_file.exists() else '❌'} {markdown_file}")
    print(f"📊 PDF: {'✅' if pdf_file.exists() else '❌'} {pdf_file}")
    print(f"🖼️  Images: {'✅' if images_dir.exists() else '❌'} {images_dir}")
    
    if markdown_file.exists():
        size_mb = markdown_file.stat().st_size / 1024 / 1024
        print(f"   • Markdown size: {size_mb:.2f} MB")
        
    if pdf_file.exists():
        size_mb = pdf_file.stat().st_size / 1024 / 1024
        print(f"   • PDF size: {size_mb:.2f} MB")
        
    if images_dir.exists():
        image_count = len(list(images_dir.glob("*.png")))
        print(f"   • Image count: {image_count} files")
    
    print("\n🚀 GENERATION COMMANDS:")
    print("• Markdown: python analysis/generate_universal_critical_restoration_report.py")
    print("• PDF: python analysis/generate_pdf_report.py")
    
    print("\n📍 OUTPUT LOCATIONS:")
    print(f"• All reports: {results_dir}/")
    print(f"• All images: {images_dir}/")

if __name__ == "__main__":
    main()
