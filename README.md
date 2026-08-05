# Transactional Fraud Detection

A complete end-to-end fraud detection project with an interactive Streamlit dashboard, trained model artifacts, and production-ready inference.

## Overview
This repository delivers a ready-to-run fraud scoring system for transaction data. It includes dataset assets, preprocessing pipelines, model artifacts, evaluation summaries, and a Streamlit app for live inference.

## Dataset
The dataset is based on the PaySim dataset from Kaggle. It was adapted for this project with a downsampled version included for efficient training and evaluation.

A live preview is available at https://transactional-fraud-detector.onrender.com.

## Key capabilities
- Real-time fraud risk scoring via a Streamlit user interface
- Support for TensorFlow/Keras and PyTorch TabNet model families
- Consistent preprocessing through serialized artifacts
- Built-in `/health` endpoint for monitoring and readiness checks
- Internal public keepalive through configured `SITE_URL`
- SEO-aware endpoints with `robots.txt` and `sitemap.xml`

## Project workflow
The project follows a standard data science workflow from data ingestion to deployment.

```text
Data source (original dataset)
         ↓
Preprocessing and feature engineering
         ↓
Model training and selection
         ↓
Serialized model artifacts and metadata
         ↓
Streamlit application loads model and preprocessing bundle
         ↓
User submits transaction input via web interface
         ↓
Model inference produces fraud risk score
         ↓
Displayed output with transaction risk assessment
```

## Key results
The table below summarizes the most important evaluation metrics for the top-performing models.

| Model | ROC-AUC | Fraud F1 | False Positives | False Negatives |
|-------|--------:|---------:|----------------:|----------------:|
| MLP | 0.9987 | 0.9686 | 34 | 43 |
| BiLSTM | 0.9975 | 0.9630 | 43 | 48 |
| GRU | 0.9974 | 0.9608 | 40 | 56 |

The MLP model is the top performer in this comparison and should be noted as the best-performing candidate for fraud detection within this project.

## Repository structure
- `app.py` — primary Streamlit application and deployment logic
- `data/` — source datasets used for model training and evaluation
- `models/` — serialized model artifacts and metadata
- `artifacts/` — preprocessing bundles and training metadata
- `notebooks/` — model training, preprocessing, and evaluation notebooks
- `results/` — evaluation outputs and prediction results
- `images/` — generated analysis plots and visualization outputs
- `render.yaml` — optional deployment configuration
- `requirements.txt` — Python package dependencies
- `README.md` — repository documentation
- `LICENSE` — license terms
- [`demo/`](demo/) — recorded deployment video demonstrating project execution
- [`documents/`](documents/) — project proposal and final report

## Installation
1. Clone the repository.
   ```bash
   git clone https://github.com/srabonti-codes/Transactional-Fraud-Detection.git
   ```
2. Change directory into the project.
   ```bash
   cd Transactional-Fraud-Detection
   ```
3. Create and activate a Python virtual environment.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Configuration
The application requires the following environment variables:

```env
PORT=8501
SITE_URL=https://your-app-url.com
```

- `PORT` defines the port used by the Streamlit server.
- `SITE_URL` is used to generate SEO routes and for internal keepalive requests.

## SEO and observability
The application exposes:

- `GET /robots.txt`
- `GET /sitemap.xml`
- Page metadata for description and canonical URL

These routes are generated dynamically based on the configured `SITE_URL`.

## Notes
- No external analytics or tracking is enabled by default.
- Health endpoint registration and keepalive behavior are handled internally by the application.

## License
This repository is distributed under the MIT License. See `LICENSE` for details.
