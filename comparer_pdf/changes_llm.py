"""
LLM-based comparison module for PDF-extracted content.
Uses LLM to intelligently !DESCRIBE! meaningful changes between two JSON files.
"""
import json
from typing import List, Dict, Any, Optional

from mvp.llm.llm_provider import OllamaProvider


class LLMVersionComparator:
    """Compares two versions of PDF-extracted JSON content using LLM for intelligent change detection."""
    
    def __init__(self, llm_provider: Optional[OllamaProvider] = None):
        """Initialize with LLM provider."""
        self.llm_provider = llm_provider or OllamaProvider()

        self.change_detection_prompt = """Du bist ein präziser Diff-Analysator. Vergleiche A (ALT) und B (NEU).

AUSGABE: Nur ein JSON-Array, nichts anderes. Keine Erklärungen davor oder danach.

IGNORIERE VÖLLIG:
- Aufzählungszeichen (-, •, *, Zahlen)
- Whitespace/Leerzeichen/Zeilenumbrüche
- Groß-/Kleinschreibung bei identischem Inhalt

MELDE NUR WICHTIGE ÄNDERUNGEN:
- Zahlenänderungen (inkl. Prozent, €, Mengen)
- Datum/Zeitänderungen
- URLs (WICHTIG: IMMER Vollständige URLs angeben, nicht nur Fragmente!)
- Bei URL-Änderungen: "URL geändert von [alte URL] zu [neue URL]"
- Negationen/Modalverben (nicht, kein, nur, muss, darf, soll, mindestens)
- Inhaltliche Wort-/Phrasenänderungen
- Regeländerungen, Strukturänderungen

FORMAT (nur dieses JSON-Array ausgeben):
[
{"typ":"SUBSTITUTION","beschreibung":"Kurze Zusammenfassung der Änderung"},
{"typ":"EINFUEGUNG","beschreibung":"Was wurde hinzugefügt"},
{"typ":"LOESCHUNG","beschreibung":"Was wurde entfernt"}
]

REGEL: Fasse zusammen was sich verändert hat. Melde die wichtigen und großen Veränderungen wie Regelveränderung, Datumveränderung, Strukturveränderung.

BEISPIEL INPUT:
ALT:
Hausaufgaben im Programmierkurs (Programmierung)
- eigenständige Auseinandersetzung mit den Konzepten
- 15% der Gesamtnote, abgabe 30.11.2023
- Hausaufgaben während des Semesters (Programmierung und Theorie)
- eigenständige Auseinandersetzung mit den Konzepten
- 35% der Gesamtnote
- Klausur am Semesterende (60min)
- 50% der Gesamtnote
- Zoom-URL für den Programmierkurs: 
- https://s.fhg.de/2023-Programmierkurs-Introprog

NEU:
Hausaufgaben im Programmierkurs (Programmierung)
- eigenständige Auseinandersetzung mit den Konzepten
- 15% der Gesamtnote, abgabe am 10.02.2025
- Hausaufgaben während des Semesters (Theorie)
- eigenständige Auseinandersetzung mit den Konzepten
- 20% der Gesamtnote
- praktischer Test am Semesterende (60min)
- 30% der Gesamtnote
- schriftlicher Test am Semesterende (60min)
- 35% der Gesamtnote
- https://s.fhg.de/2025-Programmierkurs-Introprog
- Viel Spaß!

BEISPIEL AUSGABE: 
[
{"typ":"SUBSTITUTION","beschreibung":"Hausaufgabenabgabe war am 30.11.2023, jetzt am 10.02.2025"},
{"typ":"SUBSTITUTION","beschreibung":"Hausaufgaben machten 35% der Gesamtnote, jetzt nur 20% der Gesamtnote"},
{"typ":"SUBSTITUTION","beschreibung":"Keine 60min Klausur mehr mit 50% der Gesamtnote, sondern 60min praktischer Test 30% der Gesamtnote, und 60min schriftlicher Test 35% der Gesamtnote"},
{"typ":"LOESCHUNG","beschreibung":"Zoom-URL für den Programmierkurs wurde entfernt"},
{"typ":"SUBSTITUTION","beschreibung":"Link war https://s.fhg.de/2023-Programmierkurs-Introprog, jetzt https://s.fhg.de/2025-Programmierkurs-Introprog"},
{"typ":"EINFUEGUNG","beschreibung":"Viel Spaß!"}
]

INPUT FOLGT:
"""

        self.table_comparison_prompt = """Du vergleichst zwei Tabellen: A=ALT, B=NEU. Identifiziere alle Unterschiede in Struktur und Inhalt.

        Ausgabeformat (JSON-Liste):
        [
        {"typ":"header_change","alt":["spalte1","spalte2"],"neu":["spalte1","spalte3"]},
        {"typ":"row_added","neu":["wert1","wert2","wert3"]},
        {"typ":"row_removed","alt":["wert1","wert2","wert3"]},
        {"typ":"cell_change","zeile":0,"spalte":1,"alt":"alter_wert","neu":"neuer_wert","beschreibung":"<KONTEXTUELLE BESCHREIBUNG>"},
        {"typ":"structure_change","beschreibung":"Anzahl Spalten geändert von 3 auf 4"}
        ]

        WICHTIGE Regeln:
        - Erfasse auch kleine Änderungen in Zellwerten
        - Ignoriere reine Formatierungsunterschiede
        - IMMER kontextuelle Beschreibungen verwenden basierend auf Tabellen-Header
        - Beispiel: Wenn Header "Prüfungsmodalität" und "Anteil" sind und Wert ändert von "10%" zu "20%", 
          dann beschreibe: "Der Anteil für [Prüfungsmodalität] wurde von 10% auf 20% erhöht"
        - Nutze die Spaltennamen und Zeilenkontexte für aussagekräftige Beschreibungen
        - Bei Zahlenänderungen: erkläre was die Zahlen bedeuten (z.B. Prozent, Punkte, Termine)
        - Kein Zusatztext außerhalb des JSON.

        Eingabe folgt als:
        A: <table1>
        B: <table2>"""
    
    def _clean_llm_json_response(self, response: str) -> str:
        """
        Clean and normalize LLM response to extract proper JSON array.
        Handles markdown code blocks, malformed entries, and nested structures.
        """
        # Strip whitespace
        response = response.strip()
        
        # Remove markdown code block markers
        if response.startswith('```'):
            # Find the end of the first line (language identifier)
            first_newline = response.find('\n')
            if first_newline != -1:
                response = response[first_newline + 1:]
            # Remove closing ```
            if response.endswith('```'):
                response = response[:-3]
        
        # Strip again after removing code blocks
        response = response.strip()
        
        # Remove surrounding quotes
        response = response.strip('"\'')
        
        # Try to parse and normalize the structure
        try:
            parsed = json.loads(response)
            
            # If it's already a list, normalize each entry
            if isinstance(parsed, list):
                normalized = []
                for item in parsed:
                    # Ensure proper format: typ -> type, beschreibung -> description
                    if isinstance(item, dict):
                        # Extract the actual fields we need
                        typ = item.get('typ', item.get('type', 'SUBSTITUTION'))
                        beschreibung = item.get('beschreibung', item.get('description', item.get('content', '')))
                        
                        # Skip if no description at all
                        if not beschreibung:
                            continue
                        
                        normalized.append({
                            'typ': typ,
                            'beschreibung': beschreibung
                        })
                
                # Return normalized JSON
                return json.dumps(normalized, ensure_ascii=False)
            
            # If it's a single dict, wrap in array
            elif isinstance(parsed, dict):
                typ = parsed.get('typ', parsed.get('type', 'SUBSTITUTION'))
                beschreibung = parsed.get('beschreibung', parsed.get('description', parsed.get('content', '')))
                if beschreibung:
                    return json.dumps([{'typ': typ, 'beschreibung': beschreibung}], ensure_ascii=False)
            
        except json.JSONDecodeError:
            pass  # Return as-is if can't parse
        
        return response
    
    def process_change_tracking_json(self, change_tracking_file: str) -> Dict[str, Any]:
        """
        Process change tracking JSON from align_json.py and analyze changes.
        !!! Args: change_tracking_file: Path to the JSON file generated by align_json.py !!!
        """
        print(f"🔍 Loading change tracking from {change_tracking_file}")
        
        with open(change_tracking_file, 'r', encoding='utf-8') as f:
            change_tracking = json.load(f)
        
        metadata = change_tracking.get("comparison_metadata", {})
        slides = change_tracking.get("slides", [])
        detailed_changes = []
        
        for slide in slides:
            change_type = slide.get("change_type")
            
            if change_type == "removed":
                # Slide was deleted - no llm
                detailed_changes.append({
                    "change_type": "removed",
                    "old_slide_index": slide.get("old_slide_index"),
                    "old_slide_title": slide.get("old_slide", {}).get("title", "Untitled"),
                    "old_page_number": slide.get("old_slide", {}).get("page_number"),
                    "changes": [],
                    "note": "Entire slide was removed"
                })
                
            elif change_type == "added":
                # Slide was added - no llm
                detailed_changes.append({
                    "change_type": "added",
                    "new_slide_index": slide.get("new_slide_index"),
                    "new_slide_title": slide.get("new_slide", {}).get("title", "Untitled"),
                    "new_page_number": slide.get("new_slide", {}).get("page_number"),
                    "changes": [],
                    "note": "Entire slide was added"
                })
                
            elif change_type == "unchanged":
                # Slides are identical - no llm
                detailed_changes.append({
                    "change_type": "unchanged",
                    "old_slide_index": slide.get("old_slide_index"),
                    "new_slide_index": slide.get("new_slide_index"),
                    "slide_title": slide.get("new_slide", {}).get("title", "Untitled"),
                    "similarity_score": slide.get("similarity_score"),
                    "changes": [],
                    "note": "Slides are identical, no changes detected"
                })
                
            elif change_type == "modified":
                # Slides exist in both but have changes - with llm
                print(f"🔎 Analyzing modified slide: {slide.get('old_slide', {}).get('title', 'Untitled')}")
                changes = self.analyze_slide_changes(
                    slide.get("old_slide"),
                    slide.get("new_slide")
                )
                detailed_changes.append({
                    "change_type": "modified",
                    "old_slide_index": slide.get("old_slide_index"),
                    "new_slide_index": slide.get("new_slide_index"),
                    "slide_title": slide.get("new_slide", {}).get("title", "Untitled"),
                    "similarity_score": slide.get("similarity_score"),
                    "changes": changes
                })
                
            elif change_type == "merge":
                # Multiple old slides merged into one
                print(f"🔎 Analyzing merged slide: {slide.get('new_slide', {}).get('title', 'Untitled')}")
                changes = self.analyze_merge(
                    slide.get("old_slides", []),
                    slide.get("new_slide")
                )
                detailed_changes.append({
                    "change_type": "merge",
                    "old_slide_indices": slide.get("old_slide_indices", []),
                    "new_slide_index": slide.get("new_slide_index"),
                    "new_slide_title": slide.get("new_slide", {}).get("title", "Untitled"),
                    "similarity_scores": slide.get("similarity_scores", []),
                    "changes": changes
                })
                
            elif change_type == "split":
                # One old slide split into multiple
                print(f"🔎 Analyzing split slide: {slide.get('old_slide', {}).get('title', 'Untitled')}")
                changes = self.analyze_split(
                    slide.get("old_slide"),
                    slide.get("new_slides", [])
                )
                detailed_changes.append({
                    "change_type": "split",
                    "old_slide_index": slide.get("old_slide_index"),
                    "new_slide_indices": slide.get("new_slide_indices", []),
                    "old_slide_title": slide.get("old_slide", {}).get("title", "Untitled"),
                    "similarity_scores": slide.get("similarity_scores", []),
                    "changes": changes
                })
        
        # Generate report
        report = {
            "metadata": metadata,
            "total_slides_analyzed": len(slides),
            "slides_with_content_changes": len([s for s in detailed_changes if s.get("changes")]),
            "detailed_changes": detailed_changes
        }
        
        return report
    
    def analyze_slide_changes(self, old_slide: Dict, new_slide: Dict) -> List[Dict[str, Any]]:
        """Analyze changes between two individual slides using LLM."""
        all_changes = []
        
        # Compare main content (URLs included)
        content_changes = self.detect_content_changes_llm(
            old_slide.get("content", ""),
            new_slide.get("content", "")
        )
        all_changes.extend(content_changes)
        
        # Compare tables
        table_changes = self.compare_tables_llm(
            old_slide.get("tables", []),
            new_slide.get("tables", [])
        )
        all_changes.extend(table_changes)
        
        return all_changes
    
    def analyze_merge(self, old_slides: List[Dict], new_slide: Dict) -> List[Dict[str, Any]]:
        """Analyze what happened when multiple slides were merged into one."""
        return [{
            "type": "merge_detected",
            "description": f"{len(old_slides)} slides were merged into one",
            "note": "Individual changes not analyzed for merges."
        }]
    
    def analyze_split(self, old_slide: Dict, new_slides: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze what happened when one slide was split into multiple."""
        return [{
            "type": "split_detected",
            "description": f"Slide was split into {len(new_slides)} new slides",
            "note": "Individual changes not analyzed for splits."
        }]
    
    
    def detect_content_changes_llm(self, content1: str, content2: str) -> List[Dict[str, Any]]:
        """LLM to detect meaningful changes between content."""
        if not content1 and not content2:
            return []
        
        if not content1:
            return [{'type': 'content_added', 'new_content': content2[:500]}]
        
        if not content2:
            return [{'type': 'content_removed', 'old_content': content1[:500]}]
        
        # contents are the same
        if content1 == content2:
            return []
        
        # Check if content is too large for LLM (> 8000 chars combined)
        total_length = len(content1) + len(content2)
        if total_length > 8000:
            print(f"⚠️ Content too large for LLM ({total_length} chars), truncating...")
            # Truncate but keep balance
            max_each = 4000
            content1 = content1[:max_each]
            content2 = content2[:max_each]
        
        try:
            # Prepare for LLM
            llm_input = f"A: {content1}\nB: {content2}"
            
            # Get changes from LLM
            response = self.llm_provider.chat(
                messages=[
                    {"role": "system", "content": self.change_detection_prompt},
                    {"role": "user", "content": llm_input},
                ],
                options={
                    "num_predict": 8192,  # Allow longer responses
                    "temperature": 0.7
                }
            )
            
            # Check if response is empty
            if not response or not response.strip():
                print(f"⚠️ LLM returned empty response (input length: {len(llm_input)} chars)")
                return [{'type': 'SUBSTITUTION', 'description': 'Content was modified (LLM failed to analyze)'}]
            
            # Clean up response (remove markdown, normalize structure)
            response = self._clean_llm_json_response(response)
            
            # Parse response
            try:
                llm_changes = json.loads(response)
                if isinstance(llm_changes, list):
                    # Convert to our format
                    changes = []
                    for change in llm_changes:
                        changes.append({
                            'type': change.get('typ', 'SUBSTITUTION'),
                            'description': change.get('beschreibung', '')
                        })
                    return changes
                else:
                    print(f"⚠️ Unexpected LLM response format: {response}")
                    return []
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse LLM response as JSON: {e}")
                print(f"   Response was: {response[:200]}")
                return []
        except Exception as e:
            print(f"❌ Error using LLM for content comparison: {e}")
            # Fallback
            return [{'type': 'content_change', 'description': 'Content was modified', 
                    'old_content': content1[:200] + "..." if len(content1) > 200 else content1,
                    'new_content': content2[:200] + "..." if len(content2) > 200 else content2}]
    
    def _format_change_description(self, change: Dict) -> str:
        """Format a change description based on the change type."""
        # New format: LLM generates beschreibung directly
        return change.get('beschreibung', f'Change detected: {change}')
    
    def compare_tables_llm(self, tables1: List[Dict], tables2: List[Dict]) -> List[Dict[str, Any]]:
        """Compare tables using LLM for intelligent detection."""
        changes = []
        
        # Compare each table pair using LLM
        for i, (table1, table2) in enumerate(zip(tables1, tables2)):
            table_changes = self.compare_single_table_llm(table1, table2, i)
            changes.extend(table_changes)
        
        # Handle extra tables in either version
        if len(tables1) > len(tables2):
            for i in range(len(tables2), len(tables1)):
                changes.append({
                    'type': 'table_removed',
                    'table_index': i,
                    'description': f'Table {i + 1} was removed'
                })
        elif len(tables2) > len(tables1):
            for i in range(len(tables1), len(tables2)):
                changes.append({
                    'type': 'table_added',
                    'table_index': i,
                    'description': f'Table {i + 1} was added'
                })
        
        return changes
    
    def compare_single_table_llm(self, table1: Dict, table2: Dict, table_index: int) -> List[Dict[str, Any]]:
        """Compare two individual tables using LLM."""
        changes = []
        
        try:
            # Convert tables to a comparable format
            table1_str = self._table_to_string(table1)
            table2_str = self._table_to_string(table2)
            
            # If tables are identical, no changes
            if table1_str == table2_str:
                return []
            
            # Prepare input for LLM
            llm_input = f"A: {table1_str}\nB: {table2_str}"
            
            # Get changes from LLM
            response = self.llm_provider.chat(
                messages=[
                    {"role": "system", "content": self.table_comparison_prompt},
                    {"role": "user", "content": llm_input},
                ]
            )
            
            # Clean up response (remove markdown code blocks, quotes, etc.)
            response = self._clean_llm_json_response(response)
            
            # Parse JSON response
            try:
                llm_changes = json.loads(response)
                if isinstance(llm_changes, list):
                    # Convert LLM format to our internal format
                    for change in llm_changes:
                        change_dict = {
                            'type': f"table_{change.get('typ', 'unknown_change')}",
                            'table_index': table_index,
                            'description': self._format_table_change_description(change, table_index)
                        }
                        
                        # Add specific fields based on change type
                        if 'zeile' in change:
                            change_dict['row'] = change['zeile']
                        if 'spalte' in change:
                            change_dict['column'] = change['spalte']
                        if 'alt' in change:
                            change_dict['old_value'] = change['alt']
                        if 'neu' in change:
                            change_dict['new_value'] = change['neu']
                        if 'beschreibung' in change:
                            change_dict['description'] = change['beschreibung']
                        
                        changes.append(change_dict)
                    
                    return changes
                else:
                    print(f"⚠️ Unexpected LLM table response format: {response}")
                    return []
                    
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse LLM table response as JSON: {response}")
                return []
                
        except Exception as e:
            print(f"❌ Error using LLM for table comparison: {e}")
            # Fallback: simple table change detection
            return [{
                'type': 'table_change',
                'table_index': table_index,
                'description': f'Table {table_index + 1} was modified'
            }]
    
    def _table_to_string(self, table: Dict) -> str:
        """
        Convert table dictionary to string representation for LLM comparison.
        Preserves semantic structure by using keys as field names.
        """
        lines = []
        
        # Check if table is a list of dictionaries (structured format)
        if isinstance(table, list):
            for i, row in enumerate(table):
                if isinstance(row, dict):
                    # Structured row: preserve key-value pairs
                    row_parts = []
                    for key, value in row.items():
                        row_parts.append(f"{key}={value}")
                    lines.append(f"Row {i}: {', '.join(row_parts)}")
                else:
                    lines.append(f"Row {i}: {row}")
        
        # Fallback: traditional headers + rows format
        elif isinstance(table, dict):
            headers = table.get('headers', [])
            if headers:
                lines.append("Headers: " + " | ".join(headers))
            
            rows = table.get('rows', [])
            for i, row in enumerate(rows):
                if isinstance(row, list):
                    lines.append(f"Row {i}: " + " | ".join(str(cell) for cell in row))
                elif isinstance(row, dict):
                    # Structured row
                    row_parts = [f"{k}={v}" for k, v in row.items()]
                    lines.append(f"Row {i}: {', '.join(row_parts)}")
                else:
                    lines.append(f"Row {i}: {row}")
        
        return "\n".join(lines)
    
    def _format_table_change_description(self, change: Dict, table_index: int) -> str:
        """Format a table change description."""
        typ = change.get('typ', 'unknown')
        table_num = table_index + 1
        
        if typ == 'header_change':
            return f"Table {table_num}: Headers changed from {change.get('alt')} to {change.get('neu')}"
        elif typ == 'row_added':
            return f"Table {table_num}: Row added: {change.get('neu')}"
        elif typ == 'row_removed':
            return f"Table {table_num}: Row removed: {change.get('alt')}"
        elif typ == 'cell_change':
            row = change.get('zeile', '?')
            col = change.get('spalte', '?')
            return f"Table {table_num}: Cell ({row}, {col}) changed from '{change.get('alt')}' to '{change.get('neu')}'"
        elif typ == 'structure_change':
            return f"Table {table_num}: {change.get('beschreibung', 'Structure changed')}"
        else:
            return f"Table {table_num}: {change}"


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze slide changes using LLM for intelligent change detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('json', nargs='?', help='JSON file generated by align_json.py')
    parser.add_argument('-o', '--output', help='Output file for detailed analysis report (default: changes_detailed.json)')    
    args = parser.parse_args()
    if not args.output:
        args.output = "changes_detailed.json"
        
    try:            
        # Initialize LLM provider
        llm_provider = OllamaProvider()
        
        # Process change tracking
        comparator = LLMVersionComparator(llm_provider)
        report = comparator.process_change_tracking_json(args.json)
        
        # Save report
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed change analysis saved to: {args.output}")
        print(f"   Total slides analyzed: {report['total_slides_analyzed']}")
        print(f"   Slides with content changes: {report['slides_with_content_changes']}")
        
        change_types = {}
        for slide in report['detailed_changes']:
            change_type = slide['change_type']
            change_types[change_type] = change_types.get(change_type, 0) + 1
        
        print(f"\n   Change type breakdown:")
        for change_type, count in sorted(change_types.items()):
            print(f"     {change_type}: {count}")  
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Make sure Ollama is running")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()