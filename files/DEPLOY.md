# Deploy to Streamlit Community Cloud (free)

## 1. Add files to your GitHub repo
Put these in the repo root (same folder level):
- `app.py`, `analysis.py`, `requirements.txt`
- `experiment_cohort_8000.csv`
- `appointments_300k.csv.gz`  (the app reads .gz directly)
You can also place the two data files in a `data/` folder.

## 2. Test locally (optional)
    pip install -r requirements.txt
    streamlit run app.py

## 3. Deploy
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** > deploy from GitHub.
3. Repository: `Tejaswinitata/Salon-Experiment-and-Causal-Analysis`, branch `main`, main file `app.py`.
4. Click **Deploy**. First build takes a few minutes.

## Notes
- Free apps sleep after inactivity; the first visit may take a few seconds to wake.
- Add the live link to your GitHub README and LinkedIn post.
