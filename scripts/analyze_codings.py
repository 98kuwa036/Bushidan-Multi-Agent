```python
#!/usr/bin/env python3
"""
Script to analyze contents of codings directory and generate a summary table.
"""

import os
import csv
from pathlib import Path
from datetime import datetime

def get_file_info(file_path):
    """Extract file information including size, modification time, and type."""
    stat = file_path.stat()
    size = stat.st_size
    
    # Get modified date in readable format
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    mod_date = mod_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Determine file type based on extension
    file_type = 'Unknown'
    if file_path.is_file():
        ext = file_path.suffix.lower()
        if ext in ['.py', '.js', '.html', '.css']:
            file_type = 'Code'
        elif ext in ['.md', '.txt']:
            file_type = 'Text'
        elif ext in ['.jpg', '.png', '.gif']:
            file_type = 'Image'
        elif ext == '.csv':
            file_type = 'Data'
        else:
            file_type = 'Other'
    
    return {
        'path': str(file_path),
        'type': file_type,
        'size': size,
        'modified_date': mod_date
    }

def analyze_directory(directory_path):
    """Recursively scan directory and collect file information."""
    results = []
    
    try:
        # Walk through all files and subdirectories
        for root, dirs, files in os.walk(directory_path):
            for file_name in files:
                full_path = Path(root) / file_name
                info = get_file_info(full_path)
                results.append(info)
                
    except Exception as e:
        print(f"Error scanning directory: {e}")
        return []
    
    # Sort by path to ensure consistent output
    results.sort(key=lambda x: x['path'])
    return results

def generate_csv_table(data, output_file='codings_summary.csv'):
    """Generate CSV table from data."""
    if not data:
        print("No data to write to CSV")
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['File Path', 'Type', 'Size (bytes)', 'Modified Date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow({
                'File Path': row['path'],
                'Type': row['type'],
                'Size (bytes)': str(row['size']),
                'Modified Date': row['modified_date']
            })
    
    print(f"CSV table saved to {output_file}")

def generate_markdown_table(data):
    """Generate markdown formatted table."""
    if not data:
        return "No data available"
    
    # Header
    md = "| File Path | Type | Size (bytes) | Modified Date |\n"
    md += "|-----------|------|--------------|---------------|\n"
    
    for row in data:
        md += f"| {row['path']} | {row['type']} | {row['size']} | {row['modified_date']} |\n"
    
    return md

def main():
    """Main function to analyze codings directory."""
    # Define the codings directory path
    codings_dir = Path('codings')
    
    if not codings_dir.exists():
        print(f"Directory '{codings_dir}' does not exist.")
        return
    
    print("Analyzing contents of 'codings' directory...")
    
    # Analyze directory
    file_data = analyze_directory(codings_dir)
    
    if not file_data:
        print("No files found in the codings directory.")
        return
    
    # Generate CSV table
    generate_csv_table(file_data, 'codings_summary.csv')
    
    # Generate markdown table for display
    md_table = generate_markdown_table(file_data)
    print("\nMarkdown Table:\n")
    print(md_table)

if __name__ == "__main__":
    main()
```