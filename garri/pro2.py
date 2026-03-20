import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import plotly.express as px
import shap

# --- Load model, scaler ---
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('intern.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_model_and_scaler()

#00000000000000000000000 loading the dataset 0000000000000
@st.cache_data
def load_data():
     return pd.read_csv("x.csv")
background_df = load_data()
# --- Custom CSS ---
custom_style = """
<style>
/* Sidebar background and text color */
[data-testid="stSidebar"] {
    background-color: #1E3D59; /* Dark blue */
    color: #F5F5F5; /* Light text */
}
[data-testid="stSidebar"] * {
    color: #F5F5F5 !important;
}

/* Predict button styling */
div.stButton > button:first-child {
    background-color: #FF6F61; /* Coral shade */
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
    padding: 0.6em 1.2em;
    transition: 0.3s;
}
div.stButton > button:first-child:hover {
    background-color: #E85C50;
    color: #fff;
}

/* Input table styling */
table {
    font-size: 16px;
    border-collapse: collapse;
    color: black;
    background-color:black;
    border:20px;
}
thead th {
    background-color:black;
    color: white;
    font-weight: bold;
     font-size:1.5em;
}
tbody td {
    background-color:black;
    font-weight: bold;
    font-size:2em;
}

</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# Manual mapping for Department if encoder not saved
dept_map = {
    "Admin": 0,
    "Marketing": 1,
    "Engineering": 2,
    "IT": 3,
    "Finance": 4
}

# --- Page config ---
st.set_page_config(
    page_title="Internship Retention Predictor",
    layout="wide",
    page_icon="👨‍🔬",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.markdown(
    """
<center><div class="head" style="color:orange;"border="2 solid red" > <h1> INTERN RETENTION PREDICTOR 👨‍🔬</h1></div></center>

<center><p style=" color:white;"> <b>Using ML to stabilize North West talent pipelines<br></b>
Our solutions provide HR teams with actionable insights to optimize
recruitment investments and stabilize talent pipelines </p></center>
""",
    unsafe_allow_html=True
)

# --- Sidebar inputs ---
st.sidebar.header("Selection Features")

def user_input_features():
    Department = st.sidebar.selectbox("Department", set(dept_map.keys()))
    Internship_Duration_Weeks = st.sidebar.number_input('Internship_Duration_Weeks', max_value=21, min_value=0)
    Feedback_Score = st.sidebar.number_input('Feedback_Score',max_value=10.0 , min_value=0.0, step=0.01, format="%.2f")
    Commute_Distance_km = st.sidebar.number_input('Commute_Distance_km', max_value=20.0, min_value=0.0, format="%.2f")

    data = {
        'Department': Department,
        'Internship_Duration_Weeks': Internship_Duration_Weeks,
        'Feedback_Score': Feedback_Score,
        'Commute_Distance_km': Commute_Distance_km
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Main panel ---
st.header("Your Input")
st.table(input_df.style.set_properties(**{'text-align': 'center'}))

# --- Initial Charts (before prediction) ---
st.subheader("Visualizations (Based on Inputs)")

numeric_features = input_df.drop(columns=["Department"]).melt(var_name="Feature", value_name="Value")

bar_fig = px.bar(numeric_features, x="Feature", y="Value", color="Feature", title="Input Feature Values")
st.plotly_chart(bar_fig, use_container_width=True)

pie_fig = px.pie(numeric_features, names="Feature", values="Value", title="Feature Distribution")
st.plotly_chart(pie_fig, use_container_width=True)

# --- Prediction button ---
if st.button("Predict Retention"):
    with st.spinner("Calculating prediction..."):
        time.sleep(1.5)  # Simulate processing delay

        # Encode Department
        input_df["Department"] = input_df["Department"].map(dept_map)

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)
    
    # --- Display prediction ---
    if prediction[0] == 1:
        st.success("✅ The intern is likely to return")
        outcome = "Likely to Return"
    else:
        st.error("❌ The intern is unlikely to return")
        outcome = "Unlikely to Return"

    st.markdown(
        f"<div class='prediction-result'>Predicted Retention: <b>{outcome}</b></div>",
        unsafe_allow_html=True
    )

    # --- Explainability: Feature Impact ---
    st.subheader("Feature Impact on Prediction")

    # Load some background data (for SHAP reference)
    # Ideally, use a few rows from your training dataset
    # for demo, but better to load training samples

    # explainer = shap.Explainer(model,  scaled_input, feature_names=input_df.columns)
    explainer = shap.Explainer(model,background_df,feature_names=input_df.columns)
    shap_values = explainer.shap_values(scaled_input)
    print(shap_values)

    # Convert SHAP values to DataFrame
    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Impact": shap_values[0] # explanation for first row
    })
    print(shap_df.head())

    # Bar chart of feature impacts
    impact_fig = px.bar(
        shap_df,
        x="Feature",
        y="Impact",
        color="Impact",
        title=f"Feature Impact for Prediction: {outcome}"
    )
    st.plotly_chart(impact_fig, use_container_width=True)

    # Pie chart of absolute impacts
    shap_df["AbsImpact"] = shap_df["Impact"].abs()
    pie_fig = px.pie(
        shap_df,
        names="Feature",
        values="AbsImpact",
        title="Relative Contribution of Features"
    )
    st.plotly_chart(pie_fig, use_container_width=True)


    
        # --- HR Recommendations ---#
    st.subheader("HR Recommendations")

    if outcome == "Likely to Return":
            st.markdown(
        """
            <pre> 
        <h5> - Maintain positive feedback mechanisms to keep satisfaction high.</h5>
                    . Positive feedback mechanisms: Reward good behavior.<br>
                    . Maintain : Keep it going over time.<br>
                    . Satisfaction high: Keep people happy. <br>  
            <h5>- Consider offering extended internship duration or mentorship opportunities.</h5>
                    . Exended internship duration: Give interns more time to learn.<br>
                    . Mentorship opportunities: Pair them with experienced pros.<br>
                    <b>Organisation benefits</b><br>
                    . Company turns to gain more value from interns <br>
        <h5> - Recognize achievements to reinforce commitment.</h5> 
                    . Acknowledge and reward good performance<br>
                    . Reinforce commitment : Show the intern, effort is value
            </pre>
            """,
                        unsafe_allow_html=True
            )
    else:
        st.write("""
        - Review commute distance: consider transport support or remote options.  
        - Improve feedback processes: provide coaching and regular check-ins.  
        - Adjust internship duration or workload to reduce burnout.  
        - Strengthen engagement strategies: mentorship, team integration, recognition.  
        """)

st.markdown("---")

