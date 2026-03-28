# Readability Checklist (Run Before Every Commit)

## 1. Naming and Structure
- Function and variable names are descriptive and consistent with existing modules.
- Clean text column naming is consistent (`review_clean`, not mixed variants).
- Functions are short and single-purpose.
- Public functions include type hints.

## 2. Comments and Docstrings
- Docstrings explain purpose and key assumptions.
- Comments explain why, not obvious what.
- Language is consistent (English only).
- No outdated comments after refactors.

## 3. Style Consistency
- Import order is consistent: standard library, third-party, local modules.
- Path handling uses `pathlib.Path` or shared config constants.
- String style and formatting are consistent within each file.
- No dead code, unused imports, or commented-out blocks.

## 4. Notebook Clarity
- Notebook markdown sections are concise and grammatically correct.
- Setup cells clearly show data source and output artifacts.
- Reusable logic is imported from `src/`, not duplicated in notebooks.
- Output columns and filenames match downstream modules.

## 5. Reproducibility and Safety
- Random seeds are explicit where relevant.
- Data splits are stratified for classification.
- Error messages are actionable and clear.
- File paths and expected columns are validated early.

## 6. Quick Validation
- Run lint/error check on changed Python files.
- Run a minimal sanity test for changed pipeline functions.
- Confirm notebook references still match module APIs.
- Confirm commit message describes what changed and why.
