import re
from tqdm import tqdm
import argparse

from utils import getPath, loadMarkdown, saveJson

def clearPageSplit(data):
    """
    `data`: loaded by `loadMarkdown()`, contains "\n\n<--- Page Split --->\n\n" as page split marks.
    `cleanedData`: data without page split marks, with split paragraphs naturally connected.
    """
    lines = data.split('\n')
    cleaned_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line == '<--- Page Split --->':
            # Look back for the last non-empty line before the page split
            prev_content_idx = len(cleaned_lines) - 1
            while prev_content_idx >= 0 and not cleaned_lines[prev_content_idx].strip():
                prev_content_idx -= 1
            
            # Look forward for the next non-empty line after the page split
            next_content_idx = i + 1
            while next_content_idx < len(lines) and not lines[next_content_idx].strip():
                next_content_idx += 1
            
            # Check if we have content on both sides
            has_prev_content = prev_content_idx >= 0 and cleaned_lines[prev_content_idx].strip()
            has_next_content = next_content_idx < len(lines) and lines[next_content_idx].strip()
            
            if has_prev_content and has_next_content:
                prev_line = cleaned_lines[prev_content_idx]
                next_line = lines[next_content_idx]
                
                # If both are regular content (not headings), join them
                if (not prev_line.strip().startswith('#') and 
                    not next_line.strip().startswith('#')):
                    # Remove empty lines after previous content
                    while cleaned_lines and not cleaned_lines[-1].strip():
                        cleaned_lines.pop()
                    
                    # Join with space
                    cleaned_lines[-1] = cleaned_lines[-1] + ' ' + next_line
                    
                    # Skip to after the next content line
                    i = next_content_idx
                else:
                    # Keep the paragraph break (skip the marker, keep empty lines)
                    pass
            # Otherwise just skip the page split marker
            
        else:
            cleaned_lines.append(line)
        
        i += 1
    
    return '\n'.join(cleaned_lines)

def finalize_paragraph(current_paragraph, current_section):
    """Add accumulated paragraph to current section contents"""
    if current_paragraph:
        # Join lines with space to form complete paragraph
        paragraph_text = ' '.join(line.strip() for line in current_paragraph if line.strip())
        if paragraph_text:
            current_section['contents'].append(paragraph_text)
        current_paragraph.clear()

def finalize_section(current_paragraph, current_section, sections):
    """Save current section if it has content or heading"""
    finalize_paragraph(current_paragraph, current_section)  # Make sure any pending paragraph is added
    
    if current_section['contents'] or current_section['heading']:
        if not current_section['contents']:
            current_section['contents'] = ''
        sections.append(current_section)

def mmd2Json(data, filename):
    """
    Transform cleaned markdown data to JSON structure.
    
    Args:
        data: String of cleaned markdown content
        filename: Name of the file (with or without .mmd extension)
    
    Returns:
        Dictionary with structured sections
    """
    # Remove .mmd extension if present
    file_id = filename.replace('.mmd', '')
    lines = data.split('\n')
    
    sections = []
    current_section = {
        'sid': 1,
        'heading': '',
        'heading_level': 0,
        'contents': []
    }
    
    heading_stack = []  # Stack to track heading hierarchy
    current_paragraph = []  # Buffer for accumulating paragraph lines
    
    for line in lines:
        line = line.rstrip('\n')
        
        # Check if line is a heading
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if heading_match:
            # Finalize current section before starting new one
            finalize_section(current_paragraph, current_section, sections)
            
            # Parse new heading
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2)
            
            # Update heading stack
            # Remove headings of same or lower level
            heading_stack = [h for h in heading_stack if h['level'] < level]
            heading_stack.append({'level': level, 'text': heading_text})
            
            # Create combined heading from stack
            combined_heading = ': '.join([h['text'] for h in heading_stack])
            
            # Start new section
            current_section = {
                'sid': len(sections) + 1,
                'heading': combined_heading,
                'heading_level': level,
                'contents': []
            }
        
        elif line.strip():  # Non-empty, non-heading line
            # Add to current paragraph buffer
            current_paragraph.append(line)
            
        else:  # Empty line - paragraph break
            finalize_paragraph(current_paragraph, current_section)
    
    # Add final section
    finalize_section(current_paragraph, current_section, sections)
    
    # Build final structure
    json_data = {
        'id': file_id,
        'sections': sections
    }
    
    return json_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True,
                        help="Path to a single markdown file or a directory contains multiple markdown files")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to output directory to save the transformed json file(s) (same as input location if not specified)")
    args = parser.parse_args()
    
    # Get input path
    try:
        input_path = getPath(args.path, must_exist=True)
    except Exception as e:
        print(e)
        return 1
    
    # Setup output directory
    output_dir = None
    if args.output:
        output_dir = getPath(args.output, must_exist=False)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect markdown files
    if input_path.is_file():
        if input_path.suffix not in ['.md', '.mmd', '.markdown']:
            print(f"Error: {input_path} is not a markdown file")
            return 1
        markdown_files = [input_path]
    elif input_path.is_dir():
        markdown_files = list(input_path.glob('*.md')) + \
                        list(input_path.glob('*.mmd')) + \
                        list(input_path.glob('*.markdown'))
        if not markdown_files:
            print(f"No markdown files found in {input_path}")
            return 1
    
    # Process files
    success_count = 0
    fail_count = 0
    
    for md_file in tqdm(markdown_files, desc="Processing documents"):
        try:
            data = loadMarkdown(md_file)
            cleaned_data = clearPageSplit(data)
            json_data = mmd2Json(cleaned_data, md_file.name)
            
            output_file = (output_dir or md_file.parent) / f"{md_file.stem}.json"
            saveJson(json_data, output_file)
            success_count += 1
                      
        except Exception as e:
            fail_count += 1
            print(f"\nError processing {md_file}: {e}")
    
    # Summary
    status = f"Successfully processed {success_count} file(s)"
    if fail_count > 0:
        status += f", {fail_count} failed"
    print(f"\n{status}")
    
    return 1 if fail_count > 0 else 0

if __name__ == "__main__":
    exit(main())