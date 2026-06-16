# Demo Screenshot Instructions

Screenshots must be taken from a live running instance. Do not fabricate or mock images.

## Setup (two terminals)

**Terminal 1 — backend:**
```bash
make api
# Starts FastAPI at http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

**Terminal 2 — frontend:**
```bash
make frontend
# Starts Vite dev server at http://localhost:5173
```

Open `http://localhost:5173` in your browser. Confirm the green **API online** badge appears
in the top-right of the header before taking any screenshots.

---

## Screenshot 1 — Overview / Metrics Dashboard

- Navigate to the **Overview** tab (default).
- The four metric cards should show:
  - Oracle macro-F1: **0.6728**
  - Retrieved macro-F1: **0.3887**
  - Oracle→Retrieved Gap: **0.2841**
  - Retrieval recall@10: **0.5334**
- Capture the full browser viewport at 1280 × 900 or similar.
- Save as: `docs/assets/veritas_dashboard_overview.png`

---

## Screenshot 2 — Verify Claim (full pipeline result)

- Navigate to the **Verify Claim** tab.
- Enter the following claim in the text area:
  ```
  The Apollo 11 mission landed humans on the Moon in 1969.
  ```
- Leave Top-K evidence at **5**.
- Click **Run full pipeline** and wait for results.
- The result panel should show: verdict badge, confidence %, explanation, evidence list,
  and latency breakdown chips.
- Capture after the result appears (not during loading).
- Save as: `docs/assets/veritas_verify_claim.png`

Alternative claim (simpler retrieval):
```
The Eiffel Tower is located in Paris, France.
```

---

## Screenshot 3 — Research Results

- Navigate to the **Research Results** tab.
- Scroll to the top of the page so the verifier oracle vs retrieved table is visible.
- Capture the table section (oracle F1, retrieved F1, gap) and the retrieval profile comparison.
- Save as: `docs/assets/veritas_research_results.png`

---

## Screenshot 4 — Evidence Explorer (optional)

- Navigate to the **Evidence Explorer** tab.
- Enter:
  ```
  The Eiffel Tower is located in Paris, France.
  ```
- Set Top-K to **5**, click **Retrieve evidence**.
- Capture the retrieved evidence list with scores.
- Save as: `docs/assets/veritas_evidence_explorer.png`

---

## After capturing screenshots

1. Create the assets directory if it does not exist:
   ```bash
   mkdir -p docs/assets
   ```

2. Place screenshots there:
   ```
   docs/assets/veritas_dashboard_overview.png
   docs/assets/veritas_verify_claim.png
   docs/assets/veritas_research_results.png
   docs/assets/veritas_evidence_explorer.png  (optional)
   ```

3. Add image block to `README.md` after the Key Metrics table:
   ```markdown
   ## Dashboard Screenshots

   ### Overview
   ![Veritas Overview](docs/assets/veritas_dashboard_overview.png)

   ### Verify Claim
   ![Veritas Verify Claim](docs/assets/veritas_verify_claim.png)

   ### Research Results
   ![Veritas Research Results](docs/assets/veritas_research_results.png)
   ```

4. Commit:
   ```bash
   git add docs/assets/
   git add README.md
   git commit -m "docs: add veritas dashboard screenshots"
   git push
   ```

---

## Notes

- Use a viewport of 1280 px or wider for the best table layout.
- The backend must be running for the API badge to show green; offline screenshots look broken.
- If the verifier checkpoint is missing, Verify Claim will return a 503 artifact-missing error —
  ensure `checkpoints/transformer_verifier_clean` exists before capturing.
- Do not fabricate screenshots from a mocked or local-override API.
- The `docs/assets/` directory is `.gitignore`d by default — add screenshots explicitly or
  remove the pattern from `.gitignore` if present.
