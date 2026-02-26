```markdown
# Codings Directory Analyzer

This tool analyzes the `codings/` directory and generates a summary table of all files and subdirectories.

## Usage

Run the analyzer with:

```bash
python scripts/analyze_codings.py
```

The script will generate a CSV file named `codings_summary.csv` containing information about each item in the codings directory, including:
- Name
- Path
- Type (File or Directory)
- Size (in bytes for files)
- Modification date

## Requirements

This tool uses only Python's standard library modules and requires no external dependencies.

## Output Format

The output is a CSV file with columns:
- `name`: The name of the file or directory
- `path`: Full path to the item
- `type`: Either "File" or "Directory"
- `size`: Size in bytes (0 for directories)
- `modified`: Last modification timestamp

## Example Output

| name        | path                 | type     | size  | modified           |
|-------------|----------------------|----------|-------|--------------------|
| example.py  | codings/example.py   | File     | 1234  | 2023-05-15 14:30:45 |
| subfolder   | codings/subfolder    | Directory| 0     | 2023-05-15 14:20:10 |
```