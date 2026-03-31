# ============================================================
#  FRAUD DETECTION — INTERACTIVE BROWSER DASHBOARD
#  Opens in your web browser with animations & hover effects
#  No extra libraries needed!
# ============================================================

import numpy as np
import pandas as pd
import json, webbrowser, os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score
)

print("=" * 60)
print("   FRAUD DETECTION — INTERACTIVE DASHBOARD")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1: DATA
# ─────────────────────────────────────────────────────────────
print("\n[1] Generating dataset...")
np.random.seed(42)
N_TOTAL, N_FRAUD = 10_000, 200
N_LEGIT = N_TOTAL - N_FRAUD

legit = pd.DataFrame({
    "V1": np.random.normal(0.0, 1.0, N_LEGIT),
    "V2": np.random.normal(0.0, 1.0, N_LEGIT),
    "V3": np.random.normal(0.0, 1.0, N_LEGIT),
    "V4": np.random.normal(0.0, 1.0, N_LEGIT),
    "V5": np.random.normal(0.0, 1.0, N_LEGIT),
    "Amount": np.random.exponential(50, N_LEGIT),
    "Time":   np.random.uniform(0, 172800, N_LEGIT),
    "Class":  0,
})
fraud = pd.DataFrame({
    "V1": np.random.normal(-3.0, 1.5, N_FRAUD),
    "V2": np.random.normal( 3.0, 1.5, N_FRAUD),
    "V3": np.random.normal(-2.0, 1.5, N_FRAUD),
    "V4": np.random.normal( 2.0, 1.5, N_FRAUD),
    "V5": np.random.normal(-1.5, 1.5, N_FRAUD),
    "Amount": np.random.exponential(200, N_FRAUD),
    "Time":   np.random.uniform(0, 172800, N_FRAUD),
    "Class":  1,
})
df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   ✔ {len(df):,} transactions | {N_FRAUD} fraud | {N_LEGIT:,} legitimate")

# ─────────────────────────────────────────────────────────────
# STEP 2: PREPROCESS & TRAIN
# ─────────────────────────────────────────────────────────────
print("\n[2] Preprocessing & Training...")
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
df.drop(columns=["Amount", "Time"], inplace=True)

X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train); rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test); rf_pred = rf.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]
rf_prob = rf.predict_proba(X_test)[:, 1]
lr_acc  = round(accuracy_score(y_test, lr_pred) * 100, 2)
rf_acc  = round(accuracy_score(y_test, rf_pred) * 100, 2)
lr_auc  = round(roc_auc_score(y_test, lr_prob), 4)
rf_auc  = round(roc_auc_score(y_test, rf_prob), 4)

cm_lr = confusion_matrix(y_test, lr_pred).tolist()
cm_rf = confusion_matrix(y_test, rf_pred).tolist()

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

# Sample ROC points (50 points enough for smooth curve)
idx = np.linspace(0, len(fpr_lr)-1, 50).astype(int)
roc_lr = {"fpr": fpr_lr[idx].tolist(), "tpr": tpr_lr[idx].tolist()}
idx = np.linspace(0, len(fpr_rf)-1, 50).astype(int)
roc_rf = {"fpr": fpr_rf[idx].tolist(), "tpr": tpr_rf[idx].tolist()}

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
feat_labels = importances.index.tolist()
feat_values = [round(v, 4) for v in importances.values.tolist()]

# Amount bins for histogram
legit_amounts = df[df["Class"]==0]["Amount_scaled"].values
fraud_amounts = df[df["Class"]==1]["Amount_scaled"].values

def make_hist(data, bins=30):
    counts, edges = np.histogram(data, bins=bins, range=(-2, 6))
    centers = [(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)]
    return {"x": [round(c,2) for c in centers], "y": counts.tolist()}

hist_legit = make_hist(legit_amounts)
hist_fraud = make_hist(fraud_amounts)

print(f"   ✔ LR  → Accuracy: {lr_acc}%  AUC: {lr_auc}")
print(f"   ✔ RF  → Accuracy: {rf_acc}%  AUC: {rf_auc}")

# ─────────────────────────────────────────────────────────────
# BUILD HTML DASHBOARD
# ─────────────────────────────────────────────────────────────
print("\n[3] Building interactive dashboard...")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fraud Detection Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: #0D1117;
    color: #C9D1D9;
    font-family: 'Segoe UI', sans-serif;
    min-height: 100vh;
  }}

  /* ── HEADER ── */
  .header {{
    background: linear-gradient(135deg, #161B22 0%, #1a1f2e 100%);
    border-bottom: 1px solid #21262D;
    padding: 20px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .header h1 {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #fff;
  }}
  .header h1 span {{ color: #FF4C4C; }}
  .live-badge {{
    display: flex; align-items: center; gap: 8px;
    background: #1a2a1a; border: 1px solid #2ea04326;
    padding: 6px 14px; border-radius: 999px;
    font-size: 12px; color: #3fb950;
  }}
  .pulse {{
    width: 8px; height: 8px; border-radius: 50%;
    background: #3fb950;
    animation: pulse 1.5s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50% {{ opacity:0.4; transform:scale(1.4); }}
  }}

  /* ── STAT CARDS ── */
  .stats-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    padding: 24px 32px 0;
  }}
  .stat-card {{
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    cursor: default;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
  }}
  .stat-card:hover {{
    transform: translateY(-4px);
    border-color: #58A6FF44;
    box-shadow: 0 8px 32px #58A6FF11;
  }}
  .stat-card::before {{
    content: '';
    position: absolute; top:0; left:0; right:0; height:3px;
    border-radius: 12px 12px 0 0;
  }}
  .stat-card.blue::before  {{ background: #58A6FF; }}
  .stat-card.green::before {{ background: #00E676; }}
  .stat-card.red::before   {{ background: #FF4C4C; }}
  .stat-card.gold::before  {{ background: #FFD700; }}
  .stat-label {{
    font-size: 11px; color: #8B949E;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 8px;
  }}
  .stat-value {{
    font-size: 28px; font-weight: 700;
    color: #fff; line-height: 1;
  }}
  .stat-sub {{
    font-size: 12px; color: #8B949E; margin-top: 6px;
  }}
  .stat-icon {{
    position: absolute; right: 16px; top: 16px;
    font-size: 28px; opacity: 0.15;
  }}

  /* ── CHART GRID ── */
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    padding: 20px 32px;
  }}
  .grid-3 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
    padding: 0 32px 24px;
  }}
  .card {{
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s;
  }}
  .card:hover {{ border-color: #30363D; }}
  .card-title {{
    font-size: 13px; font-weight: 600;
    color: #E6EDF3; margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
  }}
  .card-title .dot {{
    width: 8px; height: 8px; border-radius: 50%;
  }}
  canvas {{ max-height: 280px; }}

  /* ── CONFUSION MATRIX ── */
  .cm-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 4px;
  }}
  .cm-cell {{
    border-radius: 10px;
    padding: 16px 10px;
    text-align: center;
    transition: transform 0.2s, opacity 0.2s;
    cursor: default;
  }}
  .cm-cell:hover {{ transform: scale(1.06); opacity: 0.9; }}
  .cm-cell .cm-num {{
    font-size: 28px; font-weight: 700; color: #fff;
  }}
  .cm-cell .cm-lbl {{
    font-size: 10px; margin-top: 4px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }}

  /* ── FOOTER ── */
  .footer {{
    text-align: center;
    padding: 16px;
    color: #484f58;
    font-size: 12px;
    border-top: 1px solid #21262D;
  }}

  /* ── COUNT-UP ANIMATION ── */
  .count-up {{ display: inline-block; }}

  /* ── TOOLTIP OVERRIDE ── */
  .chartjs-tooltip {{ pointer-events: none; }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <h1>🛡️ FRAUD <span>DETECTION</span> ANALYTICS</h1>
  <div class="live-badge">
    <div class="pulse"></div>
    Model Active
  </div>
</div>

<!-- STAT CARDS -->
<div class="stats-row">
  <div class="stat-card blue">
    <div class="stat-label">Total Transactions</div>
    <div class="stat-value"><span class="count-up" data-target="10000">0</span></div>
    <div class="stat-sub">In dataset</div>
    <div class="stat-icon">💳</div>
  </div>
  <div class="stat-card red">
    <div class="stat-label">Fraud Cases</div>
    <div class="stat-value"><span class="count-up" data-target="200">0</span></div>
    <div class="stat-sub">2% of total</div>
    <div class="stat-icon">🚨</div>
  </div>
  <div class="stat-card gold">
    <div class="stat-label">RF Accuracy</div>
    <div class="stat-value"><span class="count-up" data-target="{rf_acc}" data-suffix="%">0</span></div>
    <div class="stat-sub">Random Forest</div>
    <div class="stat-icon">🎯</div>
  </div>
  <div class="stat-card green">
    <div class="stat-label">ROC-AUC Score</div>
    <div class="stat-value"><span class="count-up" data-target="{rf_auc}" data-decimals="4">0</span></div>
    <div class="stat-sub">Near perfect</div>
    <div class="stat-icon">📈</div>
  </div>
</div>

<!-- ROW 1: Donut + ROC -->
<div class="grid" style="padding-top:20px">
  <div class="card">
    <div class="card-title"><div class="dot" style="background:#00E676"></div>Class Distribution</div>
    <canvas id="donutChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><div class="dot" style="background:#58A6FF"></div>ROC Curve — Model Performance</div>
    <canvas id="rocChart"></canvas>
  </div>
</div>

<!-- ROW 2: Amount + Feature Importance -->
<div class="grid">
  <div class="card">
    <div class="card-title"><div class="dot" style="background:#FF4C4C"></div>Transaction Amount Distribution</div>
    <canvas id="amountChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><div class="dot" style="background:#FFD700"></div>Feature Importance (Random Forest)</div>
    <canvas id="featChart"></canvas>
  </div>
</div>

<!-- ROW 3: Confusion Matrices + Accuracy Bar -->
<div class="grid-3">
  <div class="card">
    <div class="card-title"><div class="dot" style="background:#58A6FF"></div>Confusion Matrix — Logistic Regression</div>
    <div class="cm-grid">
      <div class="cm-cell" style="background:#0d2d0d">
        <div class="cm-num">{cm_lr[0][0]}</div>
        <div class="cm-lbl" style="color:#00E676">True Negative ✓</div>
      </div>
      <div class="cm-cell" style="background:#2d0d0d">
        <div class="cm-num">{cm_lr[0][1]}</div>
        <div class="cm-lbl" style="color:#FF4C4C">False Positive ✗</div>
      </div>
      <div class="cm-cell" style="background:#2d0d0d">
        <div class="cm-num">{cm_lr[1][0]}</div>
        <div class="cm-lbl" style="color:#FF4C4C">False Negative ✗</div>
      </div>
      <div class="cm-cell" style="background:#0d2d0d">
        <div class="cm-num">{cm_lr[1][1]}</div>
        <div class="cm-lbl" style="color:#00E676">True Positive ✓</div>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="card-title"><div class="dot" style="background:#FFD700"></div>Confusion Matrix — Random Forest</div>
    <div class="cm-grid">
      <div class="cm-cell" style="background:#0d2d0d">
        <div class="cm-num">{cm_rf[0][0]}</div>
        <div class="cm-lbl" style="color:#00E676">True Negative ✓</div>
      </div>
      <div class="cm-cell" style="background:#2d0d0d">
        <div class="cm-num">{cm_rf[0][1]}</div>
        <div class="cm-lbl" style="color:#FF4C4C">False Positive ✗</div>
      </div>
      <div class="cm-cell" style="background:#2d0d0d">
        <div class="cm-num">{cm_rf[1][0]}</div>
        <div class="cm-lbl" style="color:#FF4C4C">False Negative ✗</div>
      </div>
      <div class="cm-cell" style="background:#0d2d0d">
        <div class="cm-num">{cm_rf[1][1]}</div>
        <div class="cm-lbl" style="color:#00E676">True Positive ✓</div>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="card-title"><div class="dot" style="background:#BC8CFF"></div>Model Comparison</div>
    <canvas id="accChart"></canvas>
  </div>
</div>

<div class="footer">
  Fraud Detection Analytics Dashboard &nbsp;·&nbsp; Random Forest · Logistic Regression · scikit-learn
</div>

<script>
const DATA = {{
  roc_lr: {json.dumps(roc_lr)},
  roc_rf: {json.dumps(roc_rf)},
  feat_labels: {json.dumps(feat_labels)},
  feat_values: {json.dumps(feat_values)},
  hist_legit: {json.dumps(hist_legit)},
  hist_fraud: {json.dumps(hist_fraud)},
  lr_acc: {lr_acc},
  rf_acc: {rf_acc},
  lr_auc: {lr_auc},
  rf_auc: {rf_auc},
}};

Chart.defaults.color = '#8B949E';
Chart.defaults.font.family = "'Segoe UI', sans-serif";
Chart.defaults.font.size = 11;

// ── DONUT ──────────────────────────────────────────────────
new Chart(document.getElementById('donutChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Legitimate (9,800)', 'Fraud (200)'],
    datasets: [{{
      data: [9800, 200],
      backgroundColor: ['#00E67633', '#FF4C4C33'],
      borderColor: ['#00E676', '#FF4C4C'],
      borderWidth: 2,
      hoverBackgroundColor: ['#00E67666', '#FF4C4C66'],
      hoverBorderWidth: 3,
    }}]
  }},
  options: {{
    cutout: '72%',
    animation: {{ animateRotate: true, duration: 1200 }},
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ padding: 20, usePointStyle: true }} }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.label}}: ${{ctx.parsed.toLocaleString()}} (${{(ctx.parsed/100).toFixed(1)}}%)`
        }}
      }}
    }}
  }}
}});

// ── ROC ────────────────────────────────────────────────────
new Chart(document.getElementById('rocChart'), {{
  type: 'line',
  data: {{
    datasets: [
      {{
        label: `Logistic Regression (AUC = ${{DATA.lr_auc}})`,
        data: DATA.roc_lr.fpr.map((x,i) => ({{x, y: DATA.roc_lr.tpr[i]}})),
        borderColor: '#58A6FF', backgroundColor: '#58A6FF11',
        borderWidth: 2.5, pointRadius: 0, fill: true, tension: 0.3,
      }},
      {{
        label: `Random Forest (AUC = ${{DATA.rf_auc}})`,
        data: DATA.roc_rf.fpr.map((x,i) => ({{x, y: DATA.roc_rf.tpr[i]}})),
        borderColor: '#FFD700', backgroundColor: '#FFD70011',
        borderWidth: 2.5, pointRadius: 0, fill: true, tension: 0.3,
      }},
      {{
        label: 'Random Guess (AUC = 0.5)',
        data: [{{x:0,y:0}},{{x:1,y:1}}],
        borderColor: '#484f58', borderWidth: 1,
        borderDash: [5,5], pointRadius: 0, fill: false,
      }}
    ]
  }},
  options: {{
    animation: {{ duration: 1400 }},
    scales: {{
      x: {{ type:'linear', min:0, max:1, title:{{display:true, text:'False Positive Rate'}},
             grid:{{color:'#21262D'}} }},
      y: {{ min:0, max:1, title:{{display:true, text:'True Positive Rate'}},
             grid:{{color:'#21262D'}} }},
    }},
    plugins: {{ legend: {{ position: 'bottom' }} }},
    interaction: {{ mode:'index', intersect:false }},
  }}
}});

// ── AMOUNT HISTOGRAM ───────────────────────────────────────
new Chart(document.getElementById('amountChart'), {{
  type: 'bar',
  data: {{
    labels: DATA.hist_legit.x,
    datasets: [
      {{
        label: 'Legitimate',
        data: DATA.hist_legit.y,
        backgroundColor: '#00E67644',
        borderColor: '#00E676',
        borderWidth: 1,
        borderRadius: 3,
      }},
      {{
        label: 'Fraud',
        data: DATA.hist_fraud.y,
        backgroundColor: '#FF4C4C66',
        borderColor: '#FF4C4C',
        borderWidth: 1,
        borderRadius: 3,
      }}
    ]
  }},
  options: {{
    animation: {{ duration: 1200 }},
    scales: {{
      x: {{ title:{{display:true,text:'Scaled Amount'}}, grid:{{color:'#21262D'}},
             ticks:{{ maxTicksLimit:8 }} }},
      y: {{ title:{{display:true,text:'Count'}}, grid:{{color:'#21262D'}} }},
    }},
    plugins: {{ legend:{{ position:'bottom' }} }},
    interaction: {{ mode:'index', intersect:false }},
  }}
}});

// ── FEATURE IMPORTANCE ─────────────────────────────────────
new Chart(document.getElementById('featChart'), {{
  type: 'bar',
  data: {{
    labels: DATA.feat_labels,
    datasets: [{{
      label: 'Importance Score',
      data: DATA.feat_values,
      backgroundColor: DATA.feat_values.map((v,i) =>
        i < 2 ? '#FF4C4C99' : i < 4 ? '#FFD70099' : '#58A6FF99'),
      borderColor: DATA.feat_values.map((v,i) =>
        i < 2 ? '#FF4C4C' : i < 4 ? '#FFD700' : '#58A6FF'),
      borderWidth: 1,
      borderRadius: 4,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    animation: {{ duration: 1400 }},
    scales: {{
      x: {{ title:{{display:true,text:'Importance'}}, grid:{{color:'#21262D'}} }},
      y: {{ grid:{{display:false}} }},
    }},
    plugins: {{ legend:{{display:false}} }},
  }}
}});

// ── ACCURACY + AUC GROUPED BAR ─────────────────────────────
new Chart(document.getElementById('accChart'), {{
  type: 'bar',
  data: {{
    labels: ['Logistic Reg.', 'Random Forest'],
    datasets: [
      {{
        label: 'Accuracy (%)',
        data: [DATA.lr_acc, DATA.rf_acc],
        backgroundColor: ['#58A6FF88','#FFD70088'],
        borderColor: ['#58A6FF','#FFD700'],
        borderWidth: 1.5, borderRadius: 6,
      }},
      {{
        label: 'AUC × 100',
        data: [DATA.lr_auc*100, DATA.rf_auc*100],
        backgroundColor: ['#BC8CFF88','#FF4C4C88'],
        borderColor: ['#BC8CFF','#FF4C4C'],
        borderWidth: 1.5, borderRadius: 6,
      }}
    ]
  }},
  options: {{
    animation: {{ duration: 1200 }},
    scales: {{
      y: {{ min:90, max:101, grid:{{color:'#21262D'}},
             title:{{display:true, text:'Score'}} }},
      x: {{ grid:{{display:false}} }},
    }},
    plugins: {{ legend:{{ position:'bottom' }} }},
    interaction: {{ mode:'index', intersect:false }},
  }}
}});

// ── COUNT-UP ANIMATION ─────────────────────────────────────
document.querySelectorAll('.count-up').forEach(el => {{
  const target   = parseFloat(el.dataset.target);
  const suffix   = el.dataset.suffix || '';
  const decimals = parseInt(el.dataset.decimals || '0');
  const duration = 1500;
  const step     = 16;
  const steps    = duration / step;
  let current    = 0;
  const inc      = target / steps;
  const timer = setInterval(() => {{
    current += inc;
    if (current >= target) {{ current = target; clearInterval(timer); }}
    el.textContent = decimals > 0
      ? current.toFixed(decimals) + suffix
      : Math.floor(current).toLocaleString() + suffix;
  }}, step);
}});
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────
# SAVE & OPEN IN BROWSER
# ─────────────────────────────────────────────────────────────
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_dashboard.html")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"   ✔ Saved: fraud_dashboard.html")
print("\n[4] Opening in your browser...")
webbrowser.open(f"file:///{output_file.replace(os.sep, '/')}")

print("\n" + "=" * 60)
print("   DONE! Dashboard opened in your browser.")
print("   Hover over charts for tooltips!")
print("   File saved as: fraud_dashboard.html")
print("=" * 60)
