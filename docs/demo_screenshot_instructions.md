# Demo Screenshot Instructions

Screenshots must be taken from a live running instance. Do not fabricate or mock images.

## Setup

### Terminal 1 - backend

```bash
make api
# Starts FastAPI at http://localhost:8000
```

### Terminal 2 - frontend

```bash
make frontend
# Starts Vite dev server at http://localhost:5173
```

Open `http://localhost:5173` in your browser. Confirm the backend status chip is green before taking screenshots.

## Screenshot 1 - Empty workspace

- Capture the full browser viewport with the hero section, validation snapshot, empty input, and quick-start examples visible.
- Save as `docs/assets/veritas_verify_claim.png`

## Screenshot 2 - Live verification result

- Enter:
  ```
  The Apollo 11 mission landed humans on the Moon in 1969.
  ```
- Leave the evidence depth at the default value.
- Click **Verify claim** and wait for the result.
- Capture the verdict, confidence bar, explanation, latency, and evidence cards.
- Save as `docs/assets/veritas_evidence.png`

## After capturing screenshots

1. Copy the screenshots into `docs/assets/` using the filenames above.
2. Keep the README image links pointed at those files.
3. Commit the refreshed assets together with any UI changes that caused them.

## Notes

- Use a wide viewport so the hero card and result card render side by side.
- The backend must be running for the status chip to show green.
- If the verifier checkpoint is missing, the live result will fail instead of using a mock screenshot.
