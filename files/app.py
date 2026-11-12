import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

import analysis as A

st.set_page_config(page_title="Salon Experiment & Causal Analysis", page_icon="💇", layout="wide")

PEACH, CREAM, BROWN, DARK, TAN = "#F0D9C6", "#FDF1E7", "#7A4A1E", "#3B2A1E", "#C9925A"
GITHUB = "https://github.com/Tejaswinitata/Salon-Experiment-and-Causal-Analysis"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
html, body, [class*="css"], .stApp {{ font-family:'Poppins',sans-serif; color:{DARK}; }}
.stApp {{ background:#fff; }}
header[data-testid="stHeader"], footer, #MainMenu {{ display:none; }}
.block-container {{ padding:0 0 3rem 0; max-width:100%; }}
.nav {{ background:{PEACH}; display:flex; justify-content:space-between; align-items:center; padding:22px 6vw; }}
.nav b {{ font-weight:400; letter-spacing:.12em; font-size:18px; }}
.nav span {{ font-size:13px; color:{DARK}; opacity:.75; }}
.hero {{ background:{PEACH}; padding:30px 6vw 70px; position:relative; overflow:hidden; text-align:center; }}
.hero h1 {{ font-weight:600; font-size:clamp(30px,5vw,52px); margin:10px 0 28px; }}
.ghost {{ position:absolute; right:2vw; top:0; font-size:150px; font-weight:600; color:#fff; opacity:.35;
  writing-mode:vertical-rl; line-height:1; }}
.hero-grid {{ display:flex; gap:40px; align-items:center; justify-content:center; flex-wrap:wrap; text-align:left; position:relative; }}
.hero-grid .txt {{ max-width:340px; font-size:14px; line-height:1.8; }}
.hero-grid .txt h4 {{ color:{BROWN}; font-weight:600; margin:0 0 6px; }}
.badge {{ background:{BROWN}; color:#fff; padding:12px 28px; display:inline-block; font-size:13px; margin-top:14px; letter-spacing:.06em; }}
.hcard {{ background:#fff; border-radius:200px 200px 24px 24px; padding:34px 44px; text-align:center; box-shadow:0 20px 50px rgba(122,74,30,.18); }}
.hcard .big {{ font-size:60px; font-weight:600; color:{BROWN}; line-height:1.1; }}
.hcard small {{ display:block; opacity:.7; font-size:13px; }}
.stats {{ background:{CREAM}; display:flex; justify-content:space-around; flex-wrap:wrap; gap:20px; padding:36px 6vw; text-align:center; }}
.stats b {{ font-size:30px; font-weight:500; display:block; }}
.stats span {{ font-size:13px; opacity:.7; }}
.sec {{ padding:50px 6vw 10px; text-align:center; }}
.sec h2 {{ color:{BROWN}; font-weight:500; letter-spacing:.04em; font-size:30px; margin:0 0 10px; }}
.sec p {{ max-width:640px; margin:0 auto; font-size:14px; opacity:.8; line-height:1.8; }}
.trio {{ display:flex; justify-content:space-around; gap:20px; flex-wrap:wrap; padding:30px 6vw 20px; text-align:center; }}
.trio div {{ max-width:240px; font-size:13px; opacity:.85; }}
.ico {{ width:70px; height:70px; border:2px solid {BROWN}; border-radius:50%; display:flex; align-items:center; justify-content:center;
  margin:0 auto 10px; font-size:28px; }}
.trio h5 {{ margin:0 0 6px; font-size:15px; font-weight:500; }}
.stTabs [data-baseweb="tab-list"] {{ gap:28px; justify-content:center; background:{CREAM}; padding:6px 6vw 0; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; font-size:14px; color:{DARK}; opacity:.7; }}
.stTabs [aria-selected="true"] {{ opacity:1; color:{BROWN}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{BROWN}; }}
.stTabs [data-baseweb="tab-panel"] {{ padding:28px 6vw 10px; }}
h3 {{ color:{BROWN}; font-weight:500; }}
.card {{ background:{CREAM}; border-radius:14px; padding:20px 24px; margin-bottom:14px; font-size:14px; line-height:1.7; }}
.card b {{ color:{BROWN}; }}
.kpi {{ background:{CREAM}; border-radius:14px; padding:18px; text-align:center; }}
.kpi b {{ font-size:28px; font-weight:600; color:{BROWN}; display:block; }}
.kpi span {{ font-size:12px; opacity:.7; }}
.verdict {{ border-radius:16px; padding:26px; text-align:center; background:{BROWN}; color:#fff; }}
.verdict b {{ font-size:40px; letter-spacing:.08em; }}
.foot {{ background:{CREAM}; padding:34px 6vw; margin-top:40px; font-size:13px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:16px; }}
.foot a {{ color:{BROWN}; }}
div.stButton>button, .stDownloadButton>button {{ background:{BROWN}; color:#fff; border:0; border-radius:0; padding:10px 24px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data...")
def get_data():
    om, summary = A.load_outlet_month()
    return A.load_experiment(), om, summary


exp, om, summ = get_data()
t, c = A.split(exp)
ret = A.welch_ci(t.retained_within_60d, c.retained_within_60d)
rev = A.welch_ci(t.revenue_per_visit_inr, c.revenue_per_visit_inr)
nos = A.welch_ci(t.no_show_flag, c.no_show_flag)
did_rev, did_ns = A.did(om, "revenue"), A.did(om, "no_show")
pre_est, pre_p = A.pretrend(om, "revenue")


def style(fig, h=380):
    fig.update_layout(height=h, plot_bgcolor="#fff", paper_bgcolor="#fff", font=dict(family="Poppins", color=DARK),
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(gridcolor="#f1e6dc")
    fig.update_yaxes(gridcolor="#f1e6dc")
    return fig


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><b>{value}</b><span>{label}</span></div>', unsafe_allow_html=True)


# ---------- Hero ----------
st.markdown(f"""
<div class="nav"><b>SALON LAB</b><span>Salon Experiment &amp; Causal Analysis &nbsp;|&nbsp; Synthetic case study</span></div>
<div class="hero"><div class="ghost">causal</div>
<h1>Did the change really work?</h1>
<div class="hero-grid">
 <div class="txt"><h4>Separating real impact from seasonality</h4>
 A fictional 22-outlet Coimbatore salon &amp; spa chain says its rebooking initiative improved results.
 This app tests that claim with a randomised A/B test and a difference-in-differences analysis, every estimate with a confidence interval.
 <br><span class="badge">EXPLORE THE ANALYSIS</span></div>
 <div class="hcard"><div class="big">+{ret[0]*100:.1f} pts</div><small>60-day retention lift</small>
 <small>95% CI {ret[1]*100:+.1f} to {ret[2]*100:+.1f}</small></div>
</div></div>
<div class="stats">
 <div><b>{summ['outlets']}</b><span>Outlets</span></div>
 <div><b>{summ['rows']:,}</b><span>Appointments</span></div>
 <div><b>{len(exp):,}</b><span>Clients in A/B test</span></div>
 <div><b>EXTEND</b><span>Recommendation</span></div>
</div>
<div class="sec"><h2>WHY THIS ANALYSIS?</h2>
<p>Every number below is computed live from the project data, using the same methods as the Python notebook.
<b>All data are synthetic</b>, so results demonstrate a method and are not claims about a real business.</p></div>
<div class="trio">
 <div><div class="ico">🎲</div><h5>Randomised</h5>Clients are assigned 1:1 to treatment or control, so the groups are comparable.</div>
 <div><div class="ico">📈</div><h5>Causal</h5>Difference-in-differences handles the quarter where no experiment was run.</div>
 <div><div class="ico">✅</div><h5>Honest</h5>Assumptions are tested and limits are stated next to every result.</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["A/B Design", "Experiment Results", "Difference-in-Differences", "Segments", "Variance Reduction", "Decision"])

# ---------- 1. Design ----------
with tabs[0]:
    st.markdown("### Design the test")
    l, r = st.columns([1, 1])
    with l:
        st.markdown(f"""<div class="card"><b>Hypothesis.</b> H0: the enhanced WhatsApp reminder and return-visit offer does not change
        60-day retention. H1: it increases it.<br><b>Unit:</b> identifiable client, randomised 1:1 within outlet and new/existing strata.
        Analysed by intention-to-treat.<br><b>Primary metric:</b> completed repeat visit within 60 days.<br>
        <b>Guardrails:</b> revenue per completed visit, no-show rate, contribution margin.</div>""", unsafe_allow_html=True)
        base = st.slider("Baseline retention", 0.10, 0.70, 0.34, 0.01)
        lift = st.slider("Smallest lift worth detecting (points)", 1.0, 15.0, 5.0, 0.5) / 100
        alpha = st.select_slider("Significance level (alpha)", [0.01, 0.05, 0.10], value=0.05)
        power = st.select_slider("Power", [0.70, 0.80, 0.90, 0.95], value=0.80)
        weekly = st.number_input("Eligible clients per week", 100, 10000, 1500, 100)
    with r:
        n = A.sample_size(base, min(base + lift, 0.99), alpha, power)
        weeks = int(np.ceil(2 * n / weekly))
        a, b, d = st.columns(3)
        kpi(a, f"{n:,}", "clients per arm")
        kpi(b, f"{2*n:,}", "total clients")
        kpi(d, f"{weeks} wk + 60 d", "recruitment + outcome window")
        ns = np.arange(200, 6001, 100)
        h = abs(2*np.arcsin(np.sqrt(base+lift)) - 2*np.arcsin(np.sqrt(base)))
        pw = norm.cdf(h*np.sqrt(ns/2) - norm.ppf(1-alpha/2))
        fig = go.Figure(go.Scatter(x=ns, y=pw, line=dict(color=BROWN, width=3)))
        fig.add_hline(y=power, line_dash="dash", line_color=TAN)
        fig.add_vline(x=n, line_dash="dash", line_color=TAN)
        fig.update_layout(title="Power vs clients per arm", xaxis_title="Clients per arm", yaxis_title="Power")
        st.plotly_chart(style(fig, 330), use_container_width=True)

# ---------- 2. Results ----------
with tabs[1]:
    st.markdown("### Did randomisation work?")
    a, b, d = st.columns(3)
    kpi(a, f"{len(t):,} vs {len(c):,}", "treatment vs control")
    kpi(b, f"p = {A.srm_p(exp):.3f}", "sample-ratio mismatch test")
    kpi(d, "Balanced" if A.balance(exp)["p-value"].min() > 0.05 else "Check", "pre-treatment covariates")
    bal = A.balance(exp)
    st.dataframe(bal.style.format({"Treatment": "{:.2f}", "Control": "{:.2f}", "p-value": "{:.3f}"}), hide_index=True, use_container_width=True)
    st.markdown("### Treatment effects (treatment minus control)")
    method = st.radio("Confidence interval method", ["Welch t-test", "Bootstrap (2,000 resamples)"], horizontal=True)
    metrics = {"60-day retention": ("retained_within_60d", 100, "pts"),
               "Revenue per visit": ("revenue_per_visit_inr", 1, "INR"),
               "No-show rate": ("no_show_flag", 100, "pts")}
    rows = []
    for name, (col, mult, unit) in metrics.items():
        est = A.welch_ci(t[col], c[col])[:3] if method.startswith("Welch") else A.bootstrap_ci(t[col], c[col])
        rows.append((name, t[col].mean() * (mult if mult == 100 else 1), c[col].mean() * (mult if mult == 100 else 1),
                     est[0] * mult, est[1] * mult, est[2] * mult, unit))
    res = pd.DataFrame(rows, columns=["Metric", "Treatment", "Control", "Diff", "Lo", "Hi", "Unit"])
    st.dataframe(res.style.format({"Treatment": "{:.1f}", "Control": "{:.1f}", "Diff": "{:+.2f}", "Lo": "{:+.2f}", "Hi": "{:+.2f}"}),
                 hide_index=True, use_container_width=True)
    pick = st.selectbox("Plot metric", res.Metric)
    row = res[res.Metric == pick].iloc[0]
    fig = go.Figure(go.Scatter(x=[row.Diff], y=[pick], mode="markers", marker=dict(size=14, color=BROWN),
                               error_x=dict(type="data", symmetric=False, array=[row.Hi - row.Diff], arrayminus=[row.Diff - row.Lo], color=BROWN)))
    fig.add_vline(x=0, line_dash="dash", line_color=TAN)
    fig.update_layout(title=f"Effect on {pick.lower()} with 95% CI ({row.Unit})", xaxis_title=f"Difference ({row.Unit})")
    st.plotly_chart(style(fig, 240), use_container_width=True)
    st.caption("Rule of thumb: if the interval excludes zero the effect is statistically credible; whether it is big enough to matter is a business call.")

# ---------- 3. DiD ----------
with tabs[2]:
    st.markdown("### Last quarter's rollout, with no experiment")
    st.markdown(f"""<div class="card">Outlets COI-01 to COI-11 form the treatment region; COI-12 to COI-22 are the comparison region.
    The initiative is assumed to start on <b>1 July 2026</b>. The model uses outlet and month fixed effects with standard errors clustered by outlet.</div>""", unsafe_allow_html=True)
    metric = st.radio("Outcome", ["Revenue per scheduled appointment (INR)", "No-show rate"], horizontal=True)
    key = "revenue" if metric.startswith("Revenue") else "no_show"
    tr = A.region_trend(om, key)
    fig = go.Figure()
    for reg, colr in [("Treatment Region", BROWN), ("Control Region", TAN)]:
        g = tr[tr.region == reg]
        fig.add_trace(go.Scatter(x=g.month, y=g.value, name=reg, mode="lines+markers", line=dict(color=colr, width=3)))
    fig.add_vline(x=A.CUTOFF, line_dash="dash", line_color=DARK, annotation_text="Rollout")
    fig.update_layout(title="Monthly trend by region", yaxis_title=metric)
    st.plotly_chart(style(fig, 400), use_container_width=True)
    e = did_rev if key == "revenue" else did_ns
    mult, unit = (1, "INR") if key == "revenue" else (100, "pts")
    a, b, d = st.columns(3)
    kpi(a, f"{e[0]*mult:+.1f} {unit}", "DiD effect")
    kpi(b, f"{e[1]*mult:+.1f} to {e[2]*mult:+.1f}", "95% CI")
    kpi(d, "p < 0.001" if e[3] < 0.001 else f"p = {e[3]:.3f}", "p-value")
    st.markdown(f"""<div class="card"><b>Parallel-trends check.</b> Before July the treatment region's revenue trend differed from the
    comparison region by {pre_est:.2f} INR per month (p = {pre_p:.3f}). A non-significant test is not proof the trends were parallel, so treat this
    estimate as <b>suggestive, not definitive</b>. Next steps: placebo dates, alternative control regions, and checks for concurrent promotions.</div>""", unsafe_allow_html=True)

# ---------- 4. Segments ----------
with tabs[3]:
    st.markdown("### Who gained, who lost?")
    seg = A.segments(exp)
    fig = go.Figure(go.Scatter(x=seg.Diff * 100, y=seg.Segment, mode="markers", marker=dict(size=13, color=BROWN),
                               error_x=dict(type="data", symmetric=False, array=(seg.Hi - seg.Diff) * 100, arrayminus=(seg.Diff - seg.Lo) * 100, color=BROWN)))
    fig.add_vline(x=0, line_dash="dash", line_color=TAN)
    fig.update_layout(title="Retention lift by prior-visit segment (points, 95% CI)", xaxis_title="Difference (pts)")
    st.plotly_chart(style(fig, 330), use_container_width=True)
    show = seg.assign(Treatment=seg.Treatment * 100, Control=seg.Control * 100, Diff=seg.Diff * 100, Lo=seg.Lo * 100, Hi=seg.Hi * 100)
    st.dataframe(show.style.format({"Treatment": "{:.1f}", "Control": "{:.1f}", "Diff": "{:+.1f}", "Lo": "{:+.1f}", "Hi": "{:+.1f}"}),
                 hide_index=True, use_container_width=True)
    small = seg[seg.N < 500]
    if len(small):
        st.warning(f"{', '.join(small.Segment)} has fewer than 500 clients and a very wide interval. Its negative estimate is noise-prone and should not be acted on.")
    st.caption("Segment results are exploratory. Comparing several groups raises the chance of false positives.")

# ---------- 5. Variance reduction ----------
with tabs[4]:
    st.markdown("### Could the test have been shorter?")
    adj = A.adjusted_effect(exp)
    a, b, d = st.columns(3)
    kpi(a, f"{adj['raw']*100:+.2f} pts", f"unadjusted (SE {adj['raw_se']*100:.3f})")
    kpi(b, f"{adj['adj']*100:+.2f} pts", f"adjusted (SE {adj['adj_se']*100:.3f})")
    kpi(d, f"{adj['var_reduction']*100:.2f}%", "variance reduction")
    st.markdown(f"""<div class="card">The adjustment uses pre-treatment visits, revenue and age. Here those covariates barely predict
    60-day retention, so the adjusted interval is {adj['lo']*100:+.1f} to {adj['hi']*100:+.1f} pts, essentially the same as before.
    A test with the adjusted estimate would need about <b>{adj['n_ratio']*100:.1f}%</b> of the original sample, so <b>no meaningful time is saved</b> in this data.
    Variance reduction pays off only when pre-period behaviour strongly predicts the outcome. A full CUPED version would use each client's
    pre-period outcome.</div>""", unsafe_allow_html=True)

# ---------- 6. Decision ----------
with tabs[5]:
    st.markdown("### Go, no-go or extend?")
    l, r = st.columns([1, 1])
    with l:
        thr = st.slider("Smallest retention lift that is commercially worth it (pts)", 0.0, 8.0, 5.0, 0.5,
                        help="Default 5 pts is the lift the test was powered to detect.")
        tol = st.slider("Largest acceptable rise in no-show rate (pts)", 0.0, 5.0, 2.0, 0.5)
        lo, hi = ret[1] * 100, ret[2] * 100
        guard_ok = rev[1] > 0 and nos[2] * 100 <= tol
        if hi < 0 or not guard_ok:
            verdict, why = "NO-GO", "Retention is harmed or a guardrail is breached."
        elif lo >= thr:
            verdict, why = "GO", "The whole interval clears the commercial threshold and guardrails hold."
        else:
            verdict, why = "EXTEND", "Benefit looks real, but the interval still includes lifts below your threshold. Validate before rolling out."
        st.markdown(f'<div class="verdict"><b>{verdict}</b><br>{why}</div>', unsafe_allow_html=True)
        st.caption(f"Retention CI {lo:+.1f} to {hi:+.1f} pts | revenue per visit CI {rev[1]:+.0f} to {rev[2]:+.0f} INR | no-show CI upper bound {nos[2]*100:+.1f} pts")
    with r:
        st.markdown(f"""<div class="card"><b>What could make this causal estimate wrong?</b><br>
        - Synthetic data may not mirror real client behaviour.<br>
        - DiD: non-parallel trends, region-specific shocks, spillovers, concurrent promotions.<br>
        - Walk-ins have no client ID, so retention is measured only for identifiable clients.<br>
        - Cancellations and no-shows may be misclassified at source.<br>
        - Non-compliance: some treated clients may never receive the reminder.<br>
        - Several segment comparisons raise false-positive risk.</div>""", unsafe_allow_html=True)
    st.markdown("**Conditions before a real GO:** replicate with validated data, pre-register retention as primary, set guardrail tolerances and contribution margin, run the full recruitment plus 60-day window, and run placebo and alternative-control checks.")

st.markdown(f"""<div class="foot"><div><b>Salon Experiment &amp; Causal Analysis</b><br>Yuva Intern project by Tejaswini Tata<br>
All data are synthetically generated for demonstration.</div>
<div><a href="{GITHUB}" target="_blank">View code on GitHub</a></div></div>""", unsafe_allow_html=True)
