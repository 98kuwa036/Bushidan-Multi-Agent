```python
#!/usr/bin/env python3
"""
Script to analyze the codings directory and generate a summary table.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import argparse

def get_file_info(file_path):
    """Get file information including size and modification time."""
    try:
        stat = file_path.stat()
        size = stat.st_size
        mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        return {
            'size': size,
            'mod_time': mod_time
        }
    except Exception as e:
        print(f"Error getting info for {file_path}: {e}")
        return {'size': 0, 'mod_time': 'N/A'}

def analyze_directory(directory_path):
    """Analyze directory contents and return structured data."""
    items = []
    total_size = 0
    file_count = 0
    dir_count = 0
    
    try:
        # Walk through directory recursively
        for root, dirs, files in os.walk(directory_path):
            # Process directories
            for d in dirs:
                dir_path = Path(root) / d
                items.append({
                    'name': d,
                    'path': str(dir_path.relative_to(directory_path)),
                    'type': 'Directory',
                    'size': '-',
                    'mod_time': '-'
                })
                dir_count += 1
            
            # Process files
            for f in files:
                file_path = Path(root) / f
                info = get_file_info(file_path)
                items.append({
                    'name': f,
                    'path': str(file_path.relative_to(directory_path)),
                    'type': 'File',
                    'size': info['size'],
                    'mod_time': info['mod_time']
                })
                file_count += 1
                total_size += info['size']
                
    except Exception as e:
        print(f"Error analyzing directory {directory_path}: {e}")
        return [], 0, 0, 0
    
    # Add summary row
    items.append({
        'name': 'TOTAL',
        'path': '-',
        'type': 'Summary',
        'size': total_size,
        'mod_time': '-'
    })
    
    return items, file_count, dir_count, total_size

def create_summary_table(items):
    """Create a markdown table from the item list."""
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(items)
    
    # Filter out summary row for display (we'll add it manually later)
    display_df = df[df['type'] != 'Summary'].copy()
    
    # Sort by type and then name
    display_df = display_df.sort_values(['type', 'name'])
    
    # Format size in human readable format
    def format_size(size):
        if isinstance(size, str) or size == '-':
            return size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    display_df['size'] = display_df['size'].apply(format_size)
    
    # Create markdown table with headers
    table_data = []
    for _, row in display_df.iterrows():
        table_data.append([
            row['name'],
            row['path'],
            row['type'],
            row['size'],
            row['mod_time']
        ])
    
    # Add header
    headers = ['Name', 'Path', 'Type', 'Size', 'Last Modified']
    
    # Generate markdown table
    md_table = tabulate(table_data, headers=headers, tablefmt='pipe')
    
    return md_table

def generate_summary_file(directory_path, output_file):
    """Generate the summary file with directory contents."""
    print(f"Analyzing directory: {directory_path}")
    
    items, file_count, dir_count, total_size = analyze_directory(directory_path)
    
    if not items:
        print("No items found in directory.")
        return False
    
    # Create markdown table
    md_table = create_summary_table(items)
    
    # Prepare summary statistics
    summary_stats = f"""
## Summary Statistics

- **Total Files:** {file_count}
- **Total Directories:** {dir_count}
- **Total Size:** {format_size(total_size)}
- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Directory Contents
"""

    # Write to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Codings Directory Summary\n")
            f.write(summary_stats)
            f.write(md_table)
        print(f"Summary written to {output_file}")
        return True
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False

def format_size(size):
    """Format size in human readable form."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def main():
    parser = argparse.ArgumentParser(description='Generate summary table of codings directory contents')
    parser.add_argument('--directory', '-d', default='./codings', 
                       help='Path to the codings directory (default: ./codings)')
    parser.add_argument('--output', '-o', default='codings_summary_table.md',
                       help='Output file name (default: codings_summary_table.md)')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(args.directory):
        print(f"Error: Path '{args.directory}' is not a directory.")
        sys.exit(1)
    
    # Generate summary
    success = generate_summary_file(args.directory, args.output)
    
    if success:
        print("Summary generation completed successfully!")
    else:
        print("Failed to generate summary.")
        sys.exit(1)

if __name__ == '__main__':
    main()
```