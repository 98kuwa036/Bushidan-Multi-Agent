```markdown
# Codings Directory Analyzer

This script analyzes the contents of the `codings/` directory and generates a summary table.

## Features

- Recursively scans all files in `codings/`
- Categorizes files by type (Code, Text, Image, Data, Other)
- Generates both CSV and Markdown formatted tables
- Includes file size and modification date information

## Usage

Run the script with Python 3:

```bash
python scripts/analyze_codings.py
```

This will:
1. Generate a `codings_summary.csv` file with all file information
2. Print a markdown table to stdout

## Output Files

- `codings_summary.csv`: CSV formatted summary (for spreadsheet import)
- Console output: Markdown formatted table for documentation

## Requirements

- Python 3.x
- No external dependencies required (uses only built-in modules)

## Directory Structure Assumed

The script expects a directory named `codings/` in the current working directory.
```