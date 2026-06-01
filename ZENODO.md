# Minting a Zenodo DOI for this repository

Journals (Nature Medicine, Lancet Digital Health) require code to be archived in a
citable, version-frozen form with a DOI — a live GitHub repo is not sufficient on its own.
Zenodo integrates directly with GitHub and creates an immutable snapshot + DOI for each
release. Do this **after** the public repo is pushed to GitHub.

## One-time setup
1. Sign in at https://zenodo.org with your GitHub account (or ORCID).
2. Go to **Account → GitHub** (https://zenodo.org/account/settings/github/).
3. Flip the toggle **ON** for this repository so Zenodo watches it.

## Mint the DOI
4. On GitHub, create a tagged release for the repo:
   - **Releases → Draft a new release → Choose a tag → `v1.0.0` → Publish release.**
5. Zenodo automatically captures that release and assigns a DOI within a minute or two.
6. Open the Zenodo record, copy the **concept DOI** (the one that always resolves to the
   latest version — it ends in a "all versions" badge), and verify the metadata
   (authors, title, license) — Zenodo reads `CITATION.cff` but double-check.

## Wire the DOI back in
7. Paste the DOI into:
   - `AVAILABILITY_STATEMENT.md` (Code Availability)
   - `CITATION.cff` (`doi:` field)
   - `README.md` (add a DOI badge near the top)
8. Cite the DOI in the manuscript's Code Availability statement.

## Versioning note
If you revise the code during peer review, publish a new GitHub release (`v1.1.0`, …).
Zenodo mints a new version DOI but the **concept DOI stays stable** — always cite the
concept DOI in the paper.
