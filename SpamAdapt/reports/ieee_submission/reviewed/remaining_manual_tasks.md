# Remaining Manual Tasks Before Final IEEE Submission

The reviewed LaTeX source is saved in [`reports/ieee_submission/reviewed/SpamAdapt_IEEE_Report_Reviewed.tex`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/ieee_submission/reviewed/SpamAdapt_IEEE_Report_Reviewed.tex).

Complete the following 4 remaining manual items prior to camera-ready upload:

---

### Task 1: Complete Institution and Campus Name
- **File**: `reports/ieee_submission/reviewed/SpamAdapt_IEEE_Report_Reviewed.tex`
- **Line**: 31
- **Current Content**:
  ```latex
  \textit{[Institution / University Name]} \quad \textit{[City, State, Country]} \\
  ```
- **Action**: Replace with your university or college name and city/state (e.g., `Gujarat Technological University, Ahmedabad, Gujarat, India`).

---

### Task 2: Insert Verified Author Email Addresses
- **File**: `reports/ieee_submission/reviewed/SpamAdapt_IEEE_Report_Reviewed.tex`
- **Line**: 32
- **Current Content**:
  ```latex
  \textit{Emails: [akash.uchachariya@institution.edu, umang.mori@institution.edu, keval.bamarotiya@institution.edu]}
  ```
- **Action**: Replace the bracketed list with the authentic institutional or academic email addresses of Akash Uchachariya, Umang Mori, and Keval Bamarotiya.

---

### Task 3: Insert Corresponding Author Contact
- **File**: `reports/ieee_submission/reviewed/SpamAdapt_IEEE_Report_Reviewed.tex`
- **Line**: 33
- **Current Content**:
  ```latex
  Corresponding contact: \textit{[author-email@institution.edu]}.
  ```
- **Action**: Replace with the primary corresponding author's email address.

---

### Task 4: Compile PDF in TeXLive / Overleaf & Visual Review
- **Action**:
  - The local development environment lacks `pdflatex` or a TeX distribution (reported as `NOT VERIFIED (Local pdflatex unavailable)`).
  - Import `SpamAdapt_IEEE_Report_Reviewed.tex` into [Overleaf](https://www.overleaf.com) or compile using a local TeXLive installation (`pdflatex` -> `bibtex` -> `pdflatex` -> `pdflatex`).
  - Visually verify column balancing on the final page (References section) and adjust `\enlargethispage` or `\IEEEtriggeratref` if needed for conference page limits.
