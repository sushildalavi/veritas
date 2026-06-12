# Verifier Training

- Checkpoint: `checkpoints/verifier/model.joblib`
- Classes: NOT ENOUGH INFO, REFUTED, SUPPORTED
- Sklearn version: 1.9.0
- Python version: 3.13.5
- Git commit: b1b7e1923f99107a35ba66ae91a9d9e5550972bc
- Training command: `python3 scripts/train_verifier.py`

| split | examples | accuracy | macro_f1 |
| --- | --- | --- | --- |
| train | 8 | 0.750 | 0.600 |
| validation | 5 | 0.400 | 0.286 |
| test | 6 | 0.333 | 0.250 |
