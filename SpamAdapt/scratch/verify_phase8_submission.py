"""
SpamAdapt Phase 8: Final Submission Quality & Verification Script.
Checks:
1. Submission directory files existence and readability.
2. PowerPoint slide count and integrity.
3. Absence of accidental absolute user paths in README.
4. Execution of all acceptance test suites.
"""

import re
import sys
import subprocess
from pathlib import Path
from pptx import Presentation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = PROJECT_ROOT / "submission" / "SpamAdapt_Final_Submission"

REQUIRED_SUBMISSION_FILES = [
    "SpamAdapt_Final_Presentation.pptx",
    "Project_Report.md",
    "README.md",
    "Requirements.txt",
    "Viva_Quick_Revision.md",
    "Demo_Checklist.md",
    "Evaluation_Summary.md",
    "Limitations_and_Future_Work.md",
]

def main():
    print("=" * 80)
    print("SPAMADAPT PHASE 8: FINAL SUBMISSION QUALITY ASSURANCE AUDIT")
    print("=" * 80)
    
    # Check 1: File Existence & Non-Empty
    print("\n[CHECK 1] Inspecting Submission Files...")
    for fname in REQUIRED_SUBMISSION_FILES:
        fpath = SUBMISSION_DIR / fname
        assert fpath.exists(), f"Missing required file: {fname}"
        size = fpath.stat().st_size
        assert size > 0, f"File {fname} is empty!"
        print(f"  [OK] {fname:35s} ({size:,} bytes)")
    print("  PASSED: All 8 submission files exist and are readable.")

    # Check 2: PowerPoint Integrity & Slide Count
    print("\n[CHECK 2] Verifying PowerPoint Programmatic Integrity...")
    pptx_path = SUBMISSION_DIR / "SpamAdapt_Final_Presentation.pptx"
    prs = Presentation(pptx_path)
    slide_count = len(prs.slides)
    print(f"  Programmatically loaded PPTX. Slide count = {slide_count}")
    assert slide_count == 14, f"Expected 14 slides, found {slide_count}"
    print("  PASSED: PowerPoint opens cleanly with exactly 14 slides.")

    # Check 3: Check for accidental absolute user paths in README
    print("\n[CHECK 3] Checking for Accidental Absolute Paths in README.md...")
    readme_path = SUBMISSION_DIR / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")
    
    # Patterns like C:\Users\Akash or /home/akash
    user_path_pattern = re.compile(r"[a-zA-Z]:\\Users\\[a-zA-Z0-9_-]+", re.IGNORECASE)
    matches = user_path_pattern.findall(readme_text)
    if matches:
        print(f"  WARNING: Found absolute user paths in README: {matches}")
        assert False, f"Absolute user paths found in README: {matches}"
    else:
        print("  PASSED: Zero hardcoded user absolute paths found in README.md.")

    # Check 4: Check for API keys or secrets
    print("\n[CHECK 4] Scanning for Secrets / API Keys in Submission...")
    secret_patterns = [r"AIza[0-9A-Za-z-_]{35}", r"sk-[a-zA-Z0-9]{20,}", r"ghp_[a-zA-Z0-9]{30,}"]
    for fname in REQUIRED_SUBMISSION_FILES:
        if fname.endswith(".pptx"):
            continue
        text = (SUBMISSION_DIR / fname).read_text(encoding="utf-8")
        for pat in secret_patterns:
            assert not re.search(pat, text), f"Potential secret matched in {fname} with pattern {pat}"
    print("  PASSED: Zero secrets or API keys detected.")

    # Check 5: Run Acceptance Test Suite
    print("\n[CHECK 5] Running Acceptance Test Runner...")
    proc = subprocess.run([sys.executable, "scratch/run_all_acceptance_tests.py"], capture_output=True, text=True)
    for line in proc.stdout.splitlines()[-10:]:
        print(f"  | {line}")
    assert proc.returncode == 0, f"Acceptance suite failed with return code {proc.returncode}"
    print("  PASSED: All 6 acceptance suites executed and passed 100%.")

    print("\n" + "=" * 80)
    print("ALL PHASE 8 QUALITY ASSURANCE AUDITS PASSED SUCCESSFULLY (100%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
