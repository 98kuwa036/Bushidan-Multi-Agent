```markdown
# Codings Directory Analyzer

This script analyzes the contents of a `codings` directory and generates a formatted table showing all files with their properties.

## Features

- Recursively scans the specified directory
- Displays file information including name, type, modification date, and size
- Provides summary statistics (total files and total size)
- Exports results to multiple formats:
  - CSV format for spreadsheet applications
  - Markdown table for documentation

## Requirements

- Python 3.6+
- pandas library (`pip install pandas`)

## Usage

```bash
# Analyze the default 'codings' directory
python scripts/analyze_coding_directory.py

# Analyze a specific directory
python scripts/analyze_coding_directory.py -d /path/to/directory

# Specify output directory for results
python scripts/analyze_coding_directory.py -o /output/path
```

## Output

The script will:
1. Print a formatted table to console showing all files
2. Save the data in both CSV and Markdown formats to an 'output' directory by default

## Example Output

```
=== CODINGS DIRECTORY CONTENTS ===
Name                Type    Modified           Size
main.py             .py     2023-10-15 14:30:45  2.10 KB
README.md           .md     2023-10-14 09:15:22  1.80 KB

Summary:
Total Files: 2
Total Size: 3.90 KB
```

## Notes

- The script handles errors gracefully and will skip inaccessible files
- File sizes are displayed in human-readable format (B, KB, MB, etc.)
- The default directory to analyze is 'codings' but can be overridden with the `-d` option
```