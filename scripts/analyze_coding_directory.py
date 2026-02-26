```python
#!/usr/bin/env python3
"""
Script to analyze the codings directory and generate a table of its contents.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

def get_file_info(file_path):
    """Get information about a file."""
    try:
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'path': str(file_path),
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'type': file_path.suffix.lower() if file_path.suffix else 'no extension'
        }
    except OSError as e:
        print(f"Error accessing {file_path}: {e}")
        return None

def scan_directory(directory):
    """Scan directory and collect file information."""
    files = []
    
    try:
        # Walk through the directory recursively
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                file_path = Path(root) / filename
                info = get_file_info(file_path)
                if info:
                    files.append(info)
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
        return []
    
    return files

def organize_by_type(files):
    """Organize files by type."""
    organized = {}
    for file_info in files:
        file_type = file_info['type']
        if file_type not in organized:
            organized[file_type] = []
        organized[file_type].append(file_info)
    return organized

def create_summary_table(files):
    """Create a summary table of all files."""
    df = pd.DataFrame(files)
    
    # Group by type and calculate statistics
    summary = df.groupby('type').agg({
        'size': ['count', 'sum'],
        'name': 'count'
    }).round(2)
    
    return summary

def format_size(size_bytes):
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def print_table(files):
    """Print a formatted table of file information."""
    if not files:
        print("No files found in directory.")
        return
    
    # Create DataFrame for better formatting
    df = pd.DataFrame(files)
    
    # Add human-readable size column
    df['size_readable'] = df['size'].apply(format_size)
    
    # Sort by size (largest first) and then by name
    df_sorted = df.sort_values(['size'], ascending=False).reset_index(drop=True)
    
    # Print table with pandas formatting
    print("\n=== CODINGS DIRECTORY CONTENTS ===")
    print(df_sorted[['name', 'type', 'modified', 'size_readable']].to_string(index=False))
    
    # Print summary statistics
    total_files = len(files)
    total_size = sum(f['size'] for f in files)
    
    print(f"\nSummary:")
    print(f"Total Files: {total_files}")
    print(f"Total Size: {format_size(total_size)}")

def save_tables(files, output_dir="output"):
    """Save tables to various formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(files)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "codings_directory.csv")
    df.to_csv(csv_path, index=False)
    
    # Save as Markdown table (simplified)
    md_path = os.path.join(output_dir, "codings_directory.md")
    with open(md_path, 'w') as f:
        f.write("| Name | Type | Modified | Size |\n")
        f.write("|------|------|----------|------|\n")
        for file_info in files:
            size_readable = format_size(file_info['size'])
            f.write(f"| {file_info['name']} | {file_info['type']} | {file_info['modified']} | {size_readable} |\n")
    
    print(f"\nTables saved to '{output_dir}' directory.")

def main():
    parser = argparse.ArgumentParser(description="Analyze codings directory contents.")
    parser.add_argument('--directory', '-d', default='codings', 
                       help='Directory to analyze (default: codings)')
    parser.add_argument('--output-dir', '-o', default='output',
                       help='Output directory for tables (default: output)')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return
    
    print(f"Scanning directory: {args.directory}")
    
    # Scan the directory
    files = scan_directory(args.directory)
    
    if not files:
        print("No files found in directory.")
        return
    
    # Print table to console
    print_table(files)
    
    # Save tables to output directory
    save_tables(files, args.output_dir)

if __name__ == "__main__":
    main()
```