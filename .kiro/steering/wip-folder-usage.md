---
title: WIP Folder Usage
description: Guidelines for creating temporary documents and one-off tools
inclusion: always
---

# WIP Folder Usage Guidelines

When creating temporary documents, one-off scripts, test files, or any other artifacts that are not part of the main project deliverables, always place them in a `.wip/` folder at the project root.

## Rules

1. **Always use `.wip/` for temporary artifacts:**
   - Documentation drafts
   - Test scripts
   - One-off tools
   - Experimental code
   - Build logs
   - Analysis documents
   - Setup notes
   - Any file that's helpful during development but not part of the final deliverable

2. **The `.wip/` folder is gitignored:**
   - Files in `.wip/` will not be committed to version control
   - This keeps the repository clean and focused on production code
   - You can freely create and modify files without cluttering git history

3. **Exceptions - DO NOT put in `.wip/`:**
   - README.md (project documentation)
   - Source code files that are part of the application
   - Configuration files needed for the project to run
   - Build scripts that are part of the standard workflow
   - Tests that should be version controlled

## Examples

### ✅ Good - Use `.wip/`

```
.wip/notes.md
.wip/test-script.sh
.wip/analysis.txt
.wip/SETUP_COMPLETE.md
.wip/debug-output.log
.wip/experiment.py
```

### ❌ Bad - Don't use `.wip/`

```
README.md (main project docs)
src/main.swift (source code)
package.json (config)
.github/workflows/ci.yml (CI config)
tests/unit-tests.swift (version controlled tests)
```

## Implementation

When you need to create a temporary file:

1. Check if `.wip/` exists, create it if not:

   ```bash
   mkdir -p .wip
   ```

2. Create your file in `.wip/`:

   ```bash
   # Instead of: touch SETUP_COMPLETE.md
   # Do this: touch .wip/SETUP_COMPLETE.md
   ```

3. Ensure `.wip/` is in `.gitignore`:
   ```bash
   echo ".wip/" >> .gitignore
   ```

## Benefits

- Keeps the project root clean
- Prevents accidental commits of temporary files
- Makes it clear what's production code vs. development artifacts
- Easy to clean up (just delete `.wip/` folder)
- Consistent location for all temporary files
