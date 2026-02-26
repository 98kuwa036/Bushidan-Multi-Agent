```python
#!/usr/bin/env python3
"""
Script to analyze the codings directory and generate a summary table.
"""

import os
import csv
from pathlib import Path
from datetime import datetime

def get_file_info(file_path):
    """Extract information about a file."""
    try:
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'path': str(file_path),
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'File'
        }
    except OSError as e:
        print(f"Error accessing file {file_path}: {e}")
        return None

def get_dir_info(dir_path):
    """Extract information about a directory."""
    try:
        stat = dir_path.stat()
        return {
            'name': dir_path.name,
            'path': str(dir_path),
            'size': 0,  # Directories don't have size in the same way
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'Directory'
        }
    except OSError as e:
        print(f"Error accessing directory {dir_path}: {e}")
        return None

def analyze_directory(root_dir):
    """Analyze the codings directory and collect file/directory information."""
    results = []
    
    # Walk through all subdirectories and files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Process directories
        for dirname in dirnames:
            full_path = Path(dirpath) / dirname
            info = get_dir_info(full_path)
            if info:
                results.append(info)
        
        # Process files
        for filename in filenames:
            full_path = Path(dirpath) / filename
            info = get_file_info(full_path)
            if info:
                results.append(info)
    
    return results

def write_csv_report(data, output_file):
    """Write the collected data to a CSV file."""
    if not data:
        print("No data to write.")
        return
    
    fieldnames = ['name', 'path', 'type', 'size', 'modified']
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in data:
                # Convert size to human-readable format if needed
                row['size'] = str(row['size'])  # Keep as string for simplicity
                writer.writerow(row)
        
        print(f"Summary table saved to {output_file}")
    except IOError as e:
        print(f"Error writing CSV file: {e}")

def main():
    """Main function to execute the analysis."""
    codings_dir = "codings"
    
    # Check if codings directory exists
    if not os.path.exists(codings_dir):
        print(f"Directory '{codings_dir}' does not exist.")
        return
    
    print("Analyzing codings directory...")
    data = analyze_directory(codings_dir)
    
    if not data:
        print("No files or directories found in codings/")
        return
    
    # Sort by type (directories first), then by name
    data.sort(key=lambda x: (x['type'] != 'Directory', x['name'].lower()))
    
    output_file = "codings_summary.csv"
    write_csv_report(data, output_file)
    
    print(f"Analysis complete. Found {len(data)} items.")

if __name__ == "__main__":
    main()
```