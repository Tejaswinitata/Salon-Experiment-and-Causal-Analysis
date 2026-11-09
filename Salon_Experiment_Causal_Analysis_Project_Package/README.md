# Salon Experiment & Causal Analysis — Project Package

## Important data note
All data are synthetically generated for the Yuva Intern project exercise. They are not real salon, company, or customer records. Do not describe simulated findings as actual business results.

## Files
- `appointments_300k.csv` — 300,000 appointment records across 22 fictional outlets.
- `product_sales_65k.csv` — 65,000 product sale line items.
- `experiment_cohort_8000.csv` — 8,000-client randomized A/B test cohort.
- `service_catalog.csv` — standardized service dimension.
- `stylist_dimension.csv` — stylist dimension.
- `Salon_Experiment_Causal_Analysis.ipynb` — Python analysis notebook.
- `Salon_Experiment_Causal_Analysis_Report.docx` — detailed report.
- `One_Page_Decision_Memo.docx` — leadership decision memo.
- `did_trend.png`, `ab_retention.png` — charts used in report.

## How to run
1. Install Python 3.10+.
2. Install packages: `pip install pandas numpy scipy statsmodels matplotlib jupyter`.
3. Open the notebook in Jupyter.
4. Keep the CSV files in the same folder as the notebook, or edit `DATA_DIR`.
5. Run all cells from top to bottom.

## Design choices
- Randomized experiment: client-level 1:1 assignment; primary outcome is 60-day repeat-visit retention.
- Historical quasi-experiment: 11 treatment outlets vs 11 comparison outlets; intervention assumed to begin 1 July 2026.
- Sample size: baseline retention 34%, target 39%, two-sided alpha 0.05, power 80%.
- Seed: 42.
