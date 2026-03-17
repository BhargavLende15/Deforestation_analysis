# Deforestation Analysis & Monitoring Dashboard (Indian Subcontinent)

End-to-end **beginner-friendly** project for deforestation analysis using a **Google Earth Engine exported CSV**.

## Dataset format (required)

Your CSV must contain these columns:

- `year` (integer)
- `country` (string)
- `forest_loss_area` (number; recommended unit: km²)

Example:

```csv
year,country,forest_loss_area
2015,India,1234.5
2016,India,1199.2
2015,Pakistan,210.0
```

## Project structure

```text
deforestation-project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── data_processing.py
│   ├── model.py
│   └── visualization.py
├── app/
│   └── app.py
├── dashboard/
│   └── powerbi.pbix
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Streamlit app

```bash
streamlit run app/app.py
```

## Outputs

- The app provides:
  - data cleaning + summary statistics
  - EDA plots (trend line, country comparison)
  - Linear Regression predictions for the next 5–10 years
  - model evaluation (R², MAE)

## Power BI

Open `dashboard/powerbi.pbix` in Power BI Desktop and connect it to your cleaned CSV in `data/processed/` (exported from the app if you choose).

