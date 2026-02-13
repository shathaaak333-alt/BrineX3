import streamlit as st
import pandas as pd
import numpy as np
import datetime
import io

# ==========================================
# 1. CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="BrineX: Integrated AI & Sustainability Platform",
    layout="wide",
    page_icon="🌊"
)

# Custom Header
col1, col2 = st.columns([6, 1])
with col1:
    st.title("🌊 BrineX Integrated Platform")
    st.markdown("#### AI-Driven Optimization & Sustainable Brine Management for Oman")
with col2:
    st.markdown(
        "<div style='text-align:right; font-size:24px; font-weight:bold; color:#0E5A8A;'>BrineX AI</div>",
        unsafe_allow_html=True
    )
st.markdown("---")

# ==========================================
# 2. ML BACKEND (Cached)
# ==========================================
@st.cache_resource
def train_models():
    # Helper functions
    def standardize_fit(X):
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        return (X - mu) / sigma, mu, sigma

    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    # Generate Synthetic Data
    rng = np.random.default_rng(42)
    n = 4500
    mg = rng.uniform(400, 2600, n)
    ca = rng.uniform(150, 1400, n)
    sal = rng.uniform(45000, 95000, n)
    temp = rng.uniform(15, 42, n)
    flow = rng.uniform(2000, 60000, n)
    X = np.vstack([mg, ca, sal, temp, flow]).T

    # Label Logic
    y = np.zeros(n, dtype=int)
    for i in range(n):
        sal_penalty = 0.15 if sal[i] > 85000 else 0.0
        score_mg = (mg[i] / 2000.0) - sal_penalty
        score_ca = (ca[i] / 900.0) - (0.07 if temp[i] < 20 else 0.0)
        if score_mg >= 0.85: y[i] = 1
        elif score_ca >= 0.85: y[i] = 2
        else: y[i] = 0

    # Cost Logic
    base = 0.035 * flow
    difficulty = 1.0 + 0.25 * (sal / 95000.0)
    chem_mg = (0.0009 * flow) * (mg / 1500.0)
    chem_ca = (0.0007 * flow) * (ca / 800.0)
    cost_raw = base * difficulty
    cost_raw = np.where(y == 1, cost_raw + 65 * chem_mg, cost_raw)
    cost_raw = np.where(y == 2, cost_raw + 55 * chem_ca, cost_raw)
    cost_val = cost_raw + rng.normal(0, 25, n)
    cost_val = np.maximum(cost_val, 0)

    # Train Softmax (Classifier)
    Xs, mu_c, sig_c = standardize_fit(X)
    k = 3
    Xb = np.hstack([np.ones((n, 1)), Xs])
    W = np.zeros((Xs.shape[1] + 1, k))
    Y_hot = np.zeros((n, k))
    Y_hot[np.arange(n), y] = 1.0
    
    lr = 0.14
    for _ in range(900):
        P = softmax(Xb @ W)
        grad = (Xb.T @ (P - Y_hot)) / n
        W -= lr * grad

    # Train Ridge (Regressor) for Cost
    Xs_r, mu_r, sig_r = standardize_fit(X)
    Xb_r = np.hstack([np.ones((n, 1)), Xs_r])
    lam = 2.0
    I = np.eye(Xb_r.shape[1])
    I[0, 0] = 0.0
    wR = np.linalg.solve(Xb_r.T @ Xb_r + lam * I, Xb_r.T @ cost_val)

    return (W, mu_c, sig_c), (wR, mu_r, sig_r)

# Load Models
(clf_W, clf_mu, clf_sig), (reg_w, reg_mu, reg_sig) = train_models()
LABELS = ["SKIP", "MAGNESIUM", "CALCIUM"]

def predict_single(mg, ca, sal, temp, flow):
    x = np.array([[mg, ca, sal, temp, flow]], dtype=float)
    
    # Classification
    x_s = (x - clf_mu) / clf_sig
    xb = np.hstack([np.ones((1, 1)), x_s])
    z = xb @ clf_W
    e = np.exp(z - np.max(z))
    probs = (e / np.sum(e))[0]
    mode_idx = np.argmax(probs)
    
    # Regression
    x_r = (x - reg_mu) / reg_sig
    xb_r = np.hstack([np.ones((1, 1)), x_r])
    est_cost = float(xb_r @ reg_w)
    
    return LABELS[mode_idx], probs, est_cost

def rule_decision(tds, mg_val, loc):
    if tds > 80000: return "High Salinity: Evaporation & Salt Recovery"
    elif mg_val > 1500: return "Magnesium Recovery via Chemical Precipitation"
    elif loc == "High": return "Zero Liquid Discharge (ZLD)"
    else: return "Controlled Dilution with Diffuser System"

# ==========================================
# 3. SIDEBAR INPUTS
# ==========================================
st.sidebar.header("🔬 Simulation Parameters")
analysis_mode = st.sidebar.radio("Mode", ["Single Point Simulation", "Batch File Processing"])

if analysis_mode == "Single Point Simulation":
    st.sidebar.markdown("---")
    TDS = st.sidebar.number_input("Salinity (mg/L)", 0, 150000, 65000) 
    Mg = st.sidebar.number_input("Mg²⁺ (mg/L)", 0, 10000, 1800)
    Ca = st.sidebar.number_input("Ca²⁺ (mg/L)", 0, 10000, 900)
    Temp = st.sidebar.slider("Temperature (°C)", 10, 50, 25)
    Flow = st.sidebar.number_input("Flow Rate (m³/day)", 0, 600000, 120000)
    Na = st.sidebar.number_input("Na⁺ (mg/L)", 0, 60000, 22000)
    location = st.sidebar.selectbox("Env. Sensitivity", ["Low", "Medium", "High"])

# ==========================================
# 4. MAIN LOGIC
# ==========================================

if analysis_mode == "Single Point Simulation":
    # Run Single Prediction
    rec_strategy = rule_decision(TDS, Mg, location)
    ai_mode, ai_probs, ai_cost = predict_single(Mg, Ca, TDS, Temp, Flow)
    
    mg_rec_kg = (Mg * Flow) / 1e6
    rev_mg = mg_rec_kg * 2.5 * 0.75
    
    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 AI Analysis", "📄 Report"])

    with tab1:
        st.subheader("Brine Composition & Rules")
        chart_data = pd.DataFrame({
            "Ion": ["Na⁺", "Mg²⁺", "Ca²⁺"],
            "Concentration (mg/L)": [Na, Mg, Ca]
        }).set_index("Ion")
        st.bar_chart(chart_data)
        st.info(f"**Rule-Based Strategy:** {rec_strategy}")

    with tab2:
        st.subheader("AI Prediction Model")
        c1, c2 = st.columns(2)
        c1.metric("Predicted Mode", ai_mode)
        c2.metric("Est. Cost (OMR/day)", f"{ai_cost:,.2f}")
        st.bar_chart(pd.DataFrame({"Prob": ai_probs}, index=LABELS))

    with tab3:
        st.text("Report generation ready...")

elif analysis_mode == "Batch File Processing":
    st.subheader("Batch AI Processing")
    uploaded_file = st.file_uploader("Upload Lab Data (CSV/Excel)", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("### 1. Data Preview")
            st.dataframe(df.head())

            # Define Required Columns and their Mapping
            # Key = Internal Name, Value = Expected Column Name in CSV
            req_map = {
                "Mg": "Mg_mgL", 
                "Ca": "Ca_mgL", 
                "Salinity": "Salinity_mgL", 
                "Temp": "Temp_C", 
                "Flow": "Flow_m3_day",
                "Na": "Na_mgL",
                "Location": "Location"
            }

            # Identify Missing Columns
            missing_cols = []
            final_cols = {} # Stores values or column names

            st.write("### 2. Column Mapping & Manual Input")
            st.info("Checking for required data...")

            cols_found = df.columns.tolist()
            
            # Check for each required field
            for key, csv_col in req_map.items():
                if csv_col in cols_found:
                    final_cols[key] = df[csv_col]
                    st.success(f"✅ Found {key} (Column: {csv_col})")
                else:
                    missing_cols.append(key)
            
            # If columns are missing, ask user for input
            manual_inputs = {}
            if missing_cols:
                st.warning(f"⚠️ The following columns were not found in your file: {', '.join(missing_cols)}")
                st.write("Please provide values for these missing fields (applied to all rows):")
                
                cols = st.columns(len(missing_cols))
                for i, col_name in enumerate(missing_cols):
                    with cols[i]:
                        if col_name == "Location":
                            manual_inputs[col_name] = st.selectbox(f"Select {col_name}", ["Low", "Medium", "High"])
                        else:
                            manual_inputs[col_name] = st.number_input(f"Enter {col_name}", value=0.0)
            
            # Button to Process
            if st.button("Run AI Analysis"):
                results = []
                
                # Iterate through rows
                for idx, row in df.iterrows():
                    # Extract values (either from DF or Manual Input)
                    # Helper to get value
                    def get_val(key, csv_col_name):
                        if key in manual_inputs:
                            return manual_inputs[key]
                        return row[csv_col_name]

                    v_mg = get_val("Mg", "Mg_mgL")
                    v_ca = get_val("Ca", "Ca_mgL")
                    v_sal = get_val("Salinity", "Salinity_mgL")
                    v_temp = get_val("Temp", "Temp_C")
                    v_flow = get_val("Flow", "Flow_m3_day")
                    v_na = get_val("Na", "Na_mgL")
                    v_loc = get_val("Location", "Location")

                    # AI Prediction
                    pred_mode, pred_probs, pred_cost = predict_single(v_mg, v_ca, v_sal, v_temp, v_flow)
                    
                    # Rule Decision
                    rule_res = rule_decision(v_sal, v_mg, v_loc)

                    # Append
                    results.append([
                        pred_mode, 
                        round(pred_cost, 2), 
                        rule_res,
                        round(pred_probs[0], 3), 
                        round(pred_probs[1], 3), 
                        round(pred_probs[2], 3)
                    ])

                # Create Result DataFrame
                res_df = pd.DataFrame(results, columns=["AI_Mode", "Est_Cost_OMR", "Rule_Strategy", "P_Skip", "P_Mg", "P_Ca"])
                final_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)

                st.write("### 3. Analysis Results")
                st.dataframe(final_df.head())

                # Visualize
                st.write("#### Mode Distribution")
                st.bar_chart(final_df["AI_Mode"].value_counts())

                # Download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    final_df.to_excel(writer, index=False, sheet_name='BrineX_Results')
                
                st.download_button(
                    label="📥 Download Processed Excel",
                    data=output.getvalue(),
                    file_name="BrineX_Analysis_Completed.xlsx",
                    mime="application/vnd.ms-excel"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
