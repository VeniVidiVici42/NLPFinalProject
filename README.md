# NLPFinalProject

Final project for 6.806/6.864 at MIT on question similarity. This project first implements the approach in the [following paper](https://arxiv.org/pdf/1512.05726.pdf), which substantially outperforms BM25 on the supervised AskUbuntu dataset, then works on the unsupervised Android dataset using both direct transfer and [adversarial domain adaptation](https://arxiv.org/pdf/1409.7495.pdf)

To utilize this project simply run main/main.py as-is. There is hopefully obvious flexibility in model selection and parameter choices, though some hyperparameters (e.g. learning rate) are relatively hard to access. TODO: add command line arguments for common selections.

Some metrics on the supervised AskUbuntu dataset (to be updated as hyperparameter grid search continues):

| Method | MAP (dev) | MAP (test) | MRR (dev) | MRR (test) | P@1 (dev) | P@1 (test) | P@5 (dev) | P@5 (test) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 (baseline) | 52.0 | 56.0 | 66.0 | 68.0 | 51.9 | 53.8 | 42.1 | 42.5 |
| CNN | 55.9 | 55.4 | 68.0 | 68.8 | 54.0 | 55.4 | 45.7 | 43.4 |
| LSTM | 57.0 | 56.4 | 69.3 | 69.2 | 56.0 | 54.3 | 44.4 | 42.7 |
