import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartHerd AI Dashboard", layout="wide")
st.title("🐄 SmartHerd AI Dashboard")
st.markdown("""
Monitor cattle health, detect anomalies, and get explainable AI insights in real time.
---
""")

@st.cache_resource
def load_assets():
    df = pd.read_csv('cattle_health_with_anomalies.csv')
    features = ['temperature', 'heart_rate', 'activity', 'rumination', 'milk_yield']
    model = tf.keras.models.load_model('cattle_health_model.h5')
    scaler = joblib.load('scaler.save')
    return df, features, model, scaler

df, features, model, scaler = load_assets()

st.sidebar.header("Controls")
selected_cow = st.sidebar.selectbox("Select Cow ID", df['cow_id'].unique())
show_anomalies = st.sidebar.checkbox("Show only anomalies in data table", value=False)

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🐄 Individual Cow", "📝 Predict (User Input)"])

# ===================== TAB 1: OVERVIEW =====================
with tab1:
    st.subheader("Herd Health Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cows", df['cow_id'].nunique())
    with col2:
        st.metric("Total Records", len(df))
    with col3:
        st.metric("Anomalies Detected", int(df['herd_anomaly'].sum()))

    st.markdown("### Herd Health Trends (Average per Day)")
    herd_daily = df.groupby('day')[features].mean().reset_index()
    trend_cols = st.columns(len(features))
    for i, feat in enumerate(features):
        with trend_cols[i]:
            st.line_chart(herd_daily.set_index('day')[feat], height=150, use_container_width=True)

    st.markdown("### Anomaly Distribution")
    st.bar_chart(df['herd_anomaly'].value_counts())

    st.markdown("### Data Table (First 100 Rows)")
    display_df = df[df['herd_anomaly'] == 1] if show_anomalies else df
    st.dataframe(display_df[['cow_id', 'day'] + features + ['disease', 'herd_anomaly']].head(100))

    with st.expander("What do the alerts mean?"):
        st.markdown("""
        - 🟢 **Healthy:** No action needed.
        - 🟡 **Suspicious:** Watch for signs of illness.
        - 🟠 **Sick:** Take action soon.
        - 🔴 **Very Sick:** Immediate action needed!
        """)

    with st.expander("See full raw data"):
        st.dataframe(df)

# ===================== TAB 2: INDIVIDUAL COW =====================
with tab2:
    st.subheader(f"Cow #{int(selected_cow)} - Latest Health Data")
    cow_data = df[df['cow_id'] == selected_cow].sort_values('day', ascending=False).iloc[0]
    st.write(cow_data[features + ['day']])

    # Health status and actionable advice
    X_input = scaler.transform([cow_data[features]])
    pred_prob = model.predict(X_input)[0][0]
    if pred_prob > 0.85:
        health_status = "🔴 Very Sick"
        advice = "Immediate attention needed! Contact your vet and isolate the cow if possible."
    elif pred_prob > 0.65:
        health_status = "🟠 Sick"
        advice = "Cow is likely sick. Watch closely, check for symptoms, and consult your vet."
    elif pred_prob > 0.35:
        health_status = "🟡 Suspicious"
        advice = "Cow may be unwell. Monitor eating, movement, and rumination closely."
    else:
        health_status = "🟢 Healthy"
        advice = "Cow appears healthy. Keep up the good work!"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Health Status: {health_status}")
        st.metric("Disease Probability", f"{pred_prob:.2%}")
    with col2:
        anomaly = "🚨 Yes" if cow_data['herd_anomaly'] else "✅ No"
        st.metric("Herd Anomaly", anomaly)
        st.info(advice)

    # Trend comparison: cow vs herd
    st.markdown("### Compare to Herd Average")
    cow_history = df[df['cow_id'] == selected_cow].sort_values('day')
    herd_avg = df.groupby('day')[features].mean().reset_index()
    for feat in features:
        st.line_chart(pd.DataFrame({
            f"Cow #{int(selected_cow)}": cow_history.set_index('day')[feat],
            "Herd Average": herd_avg.set_index('day')[feat]
        }))

    # Farmer-friendly SHAP explanation
    st.subheader("AI Explanation (Farmer-Friendly)")
    with st.spinner("Generating explanation..."):
        background = scaler.transform(df[features].iloc[:100])
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_input)
        shap_values = np.array(shap_values).squeeze()
        top_n = 3
        sorted_idx = np.argsort(-np.abs(shap_values))
        explanation = []
        for i in sorted_idx[:top_n]:
            impact = shap_values[i]
            feat_name = features[i].replace('_', ' ').capitalize()
            if impact > 0:
                phrase = f"{feat_name} is higher than usual, which makes the cow more likely to be sick."
            else:
                phrase = f"{feat_name} is lower than usual, which makes the cow less likely to be sick."
            explanation.append(phrase)
        st.markdown("#### Why did the AI make this prediction?")
        st.write("The AI looked at the cow's health data and found these reasons:")
        for reason in explanation:
            st.write(f"- {reason}")
        st.warning("If you see 'Sick' or 'Very Sick', take action quickly to avoid spreading illness and loss of milk.")
        st.markdown("#### SHAP Feature Impact Bar Plot:")
        fig2, ax2 = plt.subplots()
        shap.bar_plot(shap_values, feature_names=features, max_display=5, show=False)
        st.pyplot(fig2)

    # Download button for cow history
    st.download_button(
        "Download This Cow's Health History (to share with vet/family)",
        cow_history.to_csv(index=False),
        file_name=f"cow_{int(selected_cow)}_history.csv"
    )

# ===================== TAB 3: USER INPUT PREDICTION =====================
with tab3:
    st.subheader("Predict Cattle Health from User Input")
    with st.form("user_input_form"):
        temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=38.5, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=120, value=65, step=1)
        activity = st.number_input("Activity Level", min_value=0, max_value=100, value=50, step=1)
        rumination = st.number_input("Rumination Time (min)", min_value=0, max_value=1000, value=400, step=1)
        milk_yield = st.number_input("Milk Yield (litres)", min_value=0.0, max_value=50.0, value=20.0, step=0.1)
        submit = st.form_submit_button("Predict Health Status")

    if submit:
        user_features = np.array([[temperature, heart_rate, activity, rumination, milk_yield]])
        user_features_scaled = scaler.transform(user_features)
        pred_prob = model.predict(user_features_scaled)[0][0]
        if pred_prob > 0.85:
            health_status = "🔴 Very Sick"
            advice = "Immediate attention needed! Contact your vet and isolate the cow if possible."
        elif pred_prob > 0.65:
            health_status = "🟠 Sick"
            advice = "Cow is likely sick. Watch closely, check for symptoms, and consult your vet."
        elif pred_prob > 0.35:
            health_status = "🟡 Suspicious"
            advice = "Cow may be unwell. Monitor eating, movement, and rumination closely."
        else:
            health_status = "🟢 Healthy"
            advice = "Cow appears healthy. Keep up the good work!"
        st.markdown(f"### Health Status: {health_status}")
        st.info(advice)
        st.subheader(f"Prediction: {'SICK' if pred_prob > 0.5 else 'HEALTHY'} (Probability: {pred_prob:.2%})")

        with st.spinner("Generating explanation..."):
            background = scaler.transform(df[features].iloc[:100])
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(user_features_scaled)
            shap_values = np.array(shap_values).squeeze()
            top_n = 3
            sorted_idx = np.argsort(-np.abs(shap_values))
            explanation = []
            for i in sorted_idx[:top_n]:
                impact = shap_values[i]
                feat_name = features[i].replace('_', ' ').capitalize()
                if impact > 0:
                    phrase = f"{feat_name} is higher than usual, which makes the cow more likely to be sick."
                else:
                    phrase = f"{feat_name} is lower than usual, which makes the cow less likely to be sick."
                explanation.append(phrase)
            st.markdown("#### Why did the AI make this prediction?")
            st.write("The AI looked at the cow's health data and found these reasons:")
            for reason in explanation:
                st.write(f"- {reason}")
            st.warning("If you see 'Sick' or 'Very Sick', take action quickly to avoid spreading illness and loss of milk.")
            st.markdown("#### SHAP Feature Impact Bar Plot:")
            fig, ax = plt.subplots()
            shap.bar_plot(shap_values, feature_names=features, max_display=5, show=False)
            st.pyplot(fig)

st.info("Tip: Use the sidebar to select any cow and see their latest health status, anomaly detection, and AI explanation. Or use the 'Predict' tab to enter your own data!")

