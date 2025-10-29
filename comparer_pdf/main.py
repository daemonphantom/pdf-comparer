#!/usr/bin/env python3
import json
import argparse
import subprocess
from pathlib import Path

class DiffRunner:
    def __init__(self, pairs_file: str, output_dir: str = None, use_global: bool = False):
        self.pairs_file = Path(pairs_file)
        with open(self.pairs_file, 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)
        # Base output directory - will create subdirectories per pair
        self.base_output_dir = Path(output_dir) if output_dir else Path("data/comparison/pdf")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.use_global = use_global
    
    def run_module(self, module: str, args: list, timeout: int = 300):
        print(f"🔧 Running: python -m {module} {' '.join(args)}")
        result = subprocess.run(["python", "-m", module] + args, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"❌ Error running {module}:")
            print(result.stderr)
            raise RuntimeError(f"Module {module} failed")
        print(f"{module} completed")
    
    def parse_pdf(self, pdf_path: str, output_md: str):
        from capture.parsing.docling.nologo_cuda import convert_pdf_to_markdown # !!!!!!!!!!! CHANGE ACCORDING TO WHERE YOU RUN
        convert_pdf_to_markdown(pdf_path, output_md)
    
    def gen_html(self, changes_json: str, html_output: str, pair_id: str):
        with open(changes_json, 'r') as f:
            data = json.load(f)
        changes = data.get("detailed_changes", [])
        
        # Load original JSON files for side-by-side view
        alignment_json = changes_json.replace("changes_", "").replace(".json", "_alignment.json")
        old_slides = {}
        new_slides = {}
        try:
            with open(alignment_json, 'r') as f:
                alignment = json.load(f)
                for slide_info in alignment.get("slides", []):
                    if "old_slide" in slide_info:
                        idx = slide_info.get("old_slide_index")
                        old_slides[idx] = slide_info["old_slide"]
                    if "new_slide" in slide_info:
                        idx = slide_info.get("new_slide_index")
                        new_slides[idx] = slide_info["new_slide"]
        except:
            pass
        
        stats = {}
        for s in changes:
            t = s.get("change_type", "unknown")
            stats[t] = stats.get(t, 0) + 1
        
        html = f'''<!doctype html><html lang="de"><head><meta charset="utf-8"><title>{pair_id}</title><style>
            body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;line-height:1.4}}
            .container{{max-width:1400px;margin:0 auto;background:white;padding:32px;border-radius:12px}}
            .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}}
            .stat-box{{background:#f9f9f9;padding:12px;border-radius:6px;text-align:center}}
            .slide-section{{border:1px solid #e3e3e3;border-radius:8px;padding:16px;margin:16px 0}}
            .change-badge{{display:inline-block;padding:4px 12px;border-radius:12px;font-size:12px}}
            .badge-modified{{background:#fff3cd;color:#856404}}
            .badge-added{{background:#d4edda;color:#155724}}
            .badge-removed{{background:#f8d7da;color:#721c24}}
            .badge-unchanged{{background:#e7e7e7;color:#555}}
            h2{{margin:16px 0 8px 0;font-size:18px}}
            h3{{margin:12px 0 8px 0;font-size:14px;color:#333}}
            del{{text-decoration:line-through;background:#ffecec;padding:2px 4px}}
            ins{{background:#e8ffea;padding:2px 4px}}
            .changes-list{{margin:12px 0}}
            .change-item{{margin:4px 0;padding:4px 0}}
            .cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px}}
            .panel{{border:1px solid #e3e3e3;border-radius:8px;padding:12px;background:#fafafa}}
            .panel h3{{margin:0 0 8px 0}}
            .panel-content{{white-space:pre-wrap;font-size:13px;line-height:1.6}}
            </style></head><body><div class="container"><h1>{pair_id}</h1><div class="stats">'''
    
        for t, c in stats.items():
            html += f'<div class="stat-box"><div style="font-size:24px;font-weight:bold">{c}</div><div>{t}</div></div>'
        html += '</div>'
        
        for slide in changes:
            if slide.get("change_type") == "unchanged":
                continue
            ct = slide.get("change_type", "")
            title = slide.get("slide_title") or slide.get("new_slide_title") or slide.get("old_slide_title") or "Untitled"
            html += f'<div class="slide-section"><div><strong>{title}</strong> <span class="change-badge badge-{ct}">{ct}</span></div>'
            
            # Atomic changes list
            if slide.get("changes"):
                html += '<h2>Atomare Änderungen</h2><ul class="changes-list">'
                for ch in slide.get("changes", []):
                    tag = ch.get("typ") or ch.get("type") or ""
                    html += f'<li><strong>{tag}</strong>: '
                    
                    old = ch.get("alt") or ch.get("old_text")
                    new = ch.get("neu") or ch.get("new_text")
                    
                    if old and new:
                        html += f'<del>{self._esc(old)}</del> → <ins>{self._esc(new)}</ins>'
                    elif new:
                        html += f'<ins>{self._esc(new)}</ins>'
                    elif old:
                        html += f'<del>{self._esc(old)}</del>'
                    elif ch.get("description"):
                        html += self._esc(ch.get("description"))
                    html += '</li>'
                html += '</ul>'
            
            # Side-by-side view for modified slides
            if ct == "modified":
                old_idx = slide.get("old_slide_index")
                new_idx = slide.get("new_slide_index")
                old_content = old_slides.get(old_idx, {}).get("content", "")
                new_content = new_slides.get(new_idx, {}).get("content", "")
                
                if old_content or new_content:
                    html += '<h2>Seitenansicht</h2><div class="cols">'
                    html += f'<div class="panel"><h3>Alt (Seite {old_slides.get(old_idx, {}).get("page_number", "?")})</h3>'
                    html += f'<div class="panel-content">{self._format_content_with_changes(old_content, new_content, is_old=True)}</div></div>'
                    html += f'<div class="panel"><h3>Neu (Seite {new_slides.get(new_idx, {}).get("page_number", "?")})</h3>'
                    html += f'<div class="panel-content">{self._format_content_with_changes(old_content, new_content, is_old=False)}</div></div>'
                    html += '</div>'
            
            html += '</div>'
        
        html += '</div></body></html>'
        Path(html_output).write_text(html, encoding='utf-8')
    
    def _esc(self, text):
        """Escape HTML."""
        return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    
    def _format_content_with_changes(self, old_content: str, new_content: str, is_old: bool):
        """Format content with del/ins tags for side-by-side view."""
        # Simple line-by-line diff highlighting
        old_lines = old_content.split('\n')
        new_lines = new_content.split('\n')
        
        if is_old:
            result = []
            for line in old_lines:
                if line.strip() and line not in new_lines:
                    result.append(f'<del>{self._esc(line)}</del>')
                else:
                    result.append(self._esc(line))
            return '<br>'.join(result)
        else:
            result = []
            for line in new_lines:
                if line.strip() and line not in old_lines:
                    result.append(f'<ins>{self._esc(line)}</ins>')
                else:
                    result.append(self._esc(line))
            return '<br>'.join(result)
    
    def process_pair(self, pair: dict):
        pid = pair["pair_id"]
        old, new = pair["old"], pair["new"]
        if not Path(old).exists() or not Path(new).exists():
            print(f"Error: PDF files not found")
            return False
        
        # Create output directory for this pair
        output_dir = self.base_output_dir / pid
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract PDF filenames (without .pdf extension)
        old_pdf_name = Path(old).stem  # e.g., '30422_002_01_document'
        new_pdf_name = Path(new).stem  # e.g., '40280_001_02_document'
        
        # Use PDF naming convention for markdown files
        old_md = output_dir / f"{old_pdf_name}.md"
        new_md = output_dir / f"{new_pdf_name}.md"
        old_json = output_dir / f"{old_pdf_name}.json"
        new_json = output_dir / f"{new_pdf_name}.json"
        align = output_dir / f"{old_pdf_name}_{new_pdf_name}_alignment.json"
        changes = output_dir / f"changes_{old_pdf_name}_{new_pdf_name}.json"
        html = output_dir / f"diff_{pid}.html"
        
        print(f"Step 1: Parsing PDFs to markdown...")
        self.parse_pdf(old, str(old_md))
        self.parse_pdf(new, str(new_md))
        
        print(f"Step 2: Converting markdown to JSON...")
        self.run_module("capture.parsing.docling.md_processor", [str(old_md), "-o", str(old_json)])
        self.run_module("capture.parsing.docling.md_processor", [str(new_md), "-o", str(new_json)])
        
        print(f"Step 3: Aligning slides...")
        align_args = [str(old_json), str(new_json), "-j", str(align)]
        if self.use_global:
            align_args.append("--global")
            print(f"Using Greedy global match algorithm")
        else:
            print(f"Using sequential Needleman-Wunsch alignment")
        self.run_module("capture.parsing.comparer_pdf.align_json", align_args)
        
        print(f"Step 4: Detecting changes with LLM...")
        print(f"Input: {align}")
        print(f"Output: {changes}")
        # LLM analysis can take longer, timeout=1800
        self.run_module("capture.parsing.comparer_pdf.changes_llm", [str(align), "-o", str(changes)], timeout=1800)
        
        print(f"Step 5: Generating HTML...")
        if not Path(changes).exists():
            print(f"❌ Changes file not created: {changes}")
            return False
        self.gen_html(str(changes), str(html), pid)
        
        print(f"Results saved to: {output_dir}")
        return True
    
    def run(self):
        for pair in self.pairs:
            self.process_pair(pair)
        return True


def main():
    parser = argparse.ArgumentParser(description="Run PDF diff pipeline on multiple pairs")
    parser.add_argument('--pairs', required=True, help='Path to pairs.json file')
    parser.add_argument('--output-dir', default=None, help='Base output directory (default: data/comparison)')
    parser.add_argument('--global', dest='use_global', action='store_true',
                       help='Use global best-match algorithm (order-independent, better for heavily reordered slides)')
    args = parser.parse_args()
    
    runner = DiffRunner(args.pairs, args.output_dir, args.use_global)
    runner.run()

if __name__ == "__main__":
    main()
