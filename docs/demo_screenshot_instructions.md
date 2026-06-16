# Demo Screenshot Instructions

Screenshots are not auto-generated. Follow these steps to capture dashboard screenshots for the README.

## Setup

1. Start the backend (Terminal 1):
   ```bash
   make api
   ```

2. Start the frontend (Terminal 2):
   ```bash
   make frontend
   ```

3. Open `http://localhost:5173` in your browser.

## Recommended Screenshots

### 1. Overview tab — metrics cards

- Navigate to the **Overview** tab.
- The key metrics cards should show: oracle macro-F1 0.6728, retrieved macro-F1 0.3887,
  gap 0.2841, recall@10 0.5334.
- Capture the full viewport showing the metric cards and architecture diagram.
- Save as `docs/assets/veritas_overview.png`.

### 2. Verify Claim — full pipeline result

- Navigate to the **Verify Claim** tab.
- Enter: `Marie Curie won the Nobel Prize in Physics.`
- Click **Run full pipeline**.
- Wait for the result panel to appear.
- Capture the result showing verdict, explanation, evidence list, and latency breakdown.
- Save as `docs/assets/veritas_verify.png`.

### 3. Evidence Explorer

- Navigate to the **Evidence Explorer** tab.
- Enter: `The Eiffel Tower is located in Paris, France.`
- Set top-K to 5.
- Click **Retrieve evidence**.
- Capture the evidence list with scores.
- Save as `docs/assets/veritas_evidence.png`.

### 4. Research Results tab

- Navigate to the **Research Results** tab.
- Capture the verifier oracle vs retrieved table and the retrieval profile comparison table.
- Save as `docs/assets/veritas_results.png`.

## Adding to README

Once screenshots are saved in `docs/assets/`, add them to the README:

```markdown
## Screenshots

### Overview
![Veritas Overview](docs/assets/veritas_overview.png)

### Verify Claim
![Veritas Verify](docs/assets/veritas_verify.png)
```

## Notes

- Screenshots should be taken at standard viewport width (1280px or similar).
- The backend must be running for the API status badge to show "API online".
- If the verifier checkpoint is missing, the verify tab will show an error — ensure
  `checkpoints/transformer_verifier_clean` exists before taking screenshots.
- Do not fabricate screenshots from a mocked API — the demo should show real pipeline output.
