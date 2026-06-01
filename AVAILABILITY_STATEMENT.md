# Code & Data Availability Statements

> Ready-to-paste statements for the manuscript. Replace the **[bracketed placeholders]**
> (GitHub URL, Zenodo DOI, corresponding-author contact) once the repository is uploaded
> and a Zenodo archive is minted. Both statements are written to satisfy the *Nature
> Medicine* / *The Lancet Digital Health* author policies.

---

## Code Availability

All analysis code supporting the findings of this study is publicly available at
GitHub (https://github.com/yangfangs/pregnancy-causal-discovery) and archived with a permanent
Digital Object Identifier at Zenodo (**[https://doi.org/10.5281/zenodo.XXXXXXX]**).
The repository provides the complete analysis framework for data harmonization,
trimester-stratified causal discovery (PC, FCI, DirectLiNGAM, NOTEARS-MLP with bootstrap
ensembling), causal cascade identification, counterfactual intervention-window estimation
(PSW-ATT, IV-2SLS, regression discontinuity), external replication, and visualization. A
fully synthetic 500-record example dataset is included so that the entire pipeline can be
executed end-to-end without access to patient-level data. The code is released under the
MIT License. Software versions are pinned in `requirements.txt`; analyses were performed
with Python 3.10.

---

## Data Availability

The original multi-center patient-level laboratory data (845,145 clinical records from four
medical centers, 2014–2024) are not publicly available because they contain sensitive
personal health information and their sharing is restricted by Chinese healthcare data
regulations (Measures for the Administration of Health and Medical Big Data; National Health
Commission of the People's Republic of China) and by the data-use agreements of the
participating institutions. De-identified data may be made available from the corresponding
author (**[name, email]**) on reasonable request, subject to approval by the relevant
institutional ethics committees and the execution of a formal data-sharing agreement;
requests will be responded to within [e.g., 4–8 weeks]. The complete analysis framework and
a fully synthetic example dataset that reproduces the cohort's column schema and approximate
marginal distributions (containing no real patient information) are provided in the public
code repository (https://github.com/yangfangs/pregnancy-causal-discovery; **[+ Zenodo DOI]**) so that the methods can be independently
inspected and executed.

---

## Notes (delete before submission)

- **Two separate statements are required.** Nature Medicine and Lancet Digital Health both
  mandate distinct *Code availability* and *Data availability* sections — do not merge them.
- **Zenodo DOI is expected, not optional**, for the archived code snapshot. Mint it *after*
  pushing to GitHub (see `ZENODO.md`), then fill the DOI in above and in `README.md` /
  `CITATION.cff`.
- **"On reasonable request" alone is increasingly scrutinized.** Strengthen it as above by
  naming (a) who to contact, (b) the approval gate (ethics committee + data-sharing
  agreement), and (c) a response timeframe.
- If the journal requires a **structured reporting checklist** (e.g., Nature's Reporting
  Summary), the synthetic dataset lets reviewers run the framework end-to-end, satisfying
  the "code is executable / methods are reproducible" requirement. If a reviewer or editor
  specifically asks for the study's *exact* pre-computed outputs or figure-generation code
  (intentionally excluded from this framework release), provide them privately on request.
