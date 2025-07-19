# MLOps Assignment 2 – Digit Classification Pipeline

This repository contains the implementation of a complete MLOps pipeline for digit classification using Logistic Regression on the `sklearn.datasets.load_digits` dataset. The goal was to integrate modular ML code with continuous integration workflows using GitHub Actions. The project follows best practices including parameterized training using JSON config files, unit testing with Pytest, and multi-job CI workflows with artifact sharing.

The workflow is structured into three phases: model training, unit testing, and inference — each executed in its own GitHub Actions job. The model is trained using configurable hyperparameters, validated through automated tests, and used to generate predictions in a separate job using the saved model artifact. This pipeline ensures reproducibility, quality assurance, and end-to-end automation for a simple ML task.

The structure includes modular Python scripts (`train.py`, `inference.py`, and `utils.py`) and corresponding workflows (`train.yml`, `test.yml`, and `inference.yml`) under `.github/workflows`. Each phase is implemented in a separate branch (`classification_branch`, `test_branch`, and `inference_branch`) to maintain clarity and version isolation, as per the assignment guidelines.
