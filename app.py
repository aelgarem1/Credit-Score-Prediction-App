import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder as labael_enc
# =========================================
# 1️⃣ Sidebar Information
# =========================================
with st.sidebar:
    st.image("my photo.jpg", width=130)
    st.markdown("### Amira Ali")
    st.markdown("**Data Analyst | BI Developer**")
    st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/yourprofile)")
    selected_tab = st.radio(
        "Navigation",
        [
            "Data Overview",
            "EDA & Feature Engineering",
            "Model Comparison",
            "Feature Importance",
            
            "Prediction"
        ]
    )

# =========================================
# 2️⃣ Load Data and Models
# =========================================

def load_data():

    df_cleaned = pd.read_csv("df_cleaned.csv") 
    df = pd.read_csv("train.csv") 
    df_customer = pd.read_csv("df_customer.csv", index_col=[0]) 
    return df, df_cleaned, df_customer

df, df_cleaned, df_customer = load_data()


na = pd.read_csv("nonaggregated_model_comparison_results1.csv")
a = pd.read_csv("aggregated_model_comparison_results.csv")

best_model = joblib.load("aggregated_best_credit_score_model.pkl")



# =========================================
# 3️⃣ Data Overview
# =========================================
if selected_tab == "Data Overview":
    st.header("📊 Data Overview")
    st.write("Basic overview of the dataset used for credit score prediction.")

    st.write("**Dataset Shape:**", df.shape)
    st.write("**Column Names:**", list(df.columns))

    st.subheader("Missing Values per Column")
    st.dataframe(df.isna().sum().to_frame("Missing Count"))

    st.subheader("Sample Data")
    st.dataframe(df_cleaned.head())


# =========================================
# 4️⃣ EDA Tab
# =========================================
elif selected_tab == "EDA & Feature Engineering":
    st.header("🔍 Exploratory Data Analysis")

    st.write("""
    During the EDA phase, I checked:
    - Missing values and data types  
    - Cardinality of categorical variables  
    - Class imbalance in the target  
    - Summary statistics for numeric features  
    - Distribution of key numerical columns
    """)
    df_cleaned = df_cleaned.drop('Customer_ID', axis =1)
    numeric_col = st.selectbox("Select a numeric column to visualize:", df_cleaned.select_dtypes("number").columns)
    fig = px.histogram(df_cleaned, x=numeric_col, nbins=30, title=f"Distribution of {numeric_col}")
    st.plotly_chart(fig)

    target_col = st.selectbox("Select a categorical column to compare with Credit_Score:", df_cleaned.select_dtypes("object").drop('Credit_Score', axis =1).columns)
    fig = px.box(df_cleaned, x="Credit_Score", y=target_col, color="Credit_Score",
                 title=f"{target_col} distribution across Credit Score")
    st.plotly_chart(fig)


    # Correlation heatmap
    corr = df_customer.select_dtypes("number").corr()
    fig_corr = px.imshow(corr, text_auto=False, title="Feature Correlation Heatmap",
                         color_continuous_scale="RdBu", zmin=-1, zmax=1)
    st.plotly_chart(fig_corr)

    # Credit Score Distribution
    fig_dist = px.histogram(df, x="Credit_Score", color="Credit_Score",
                            title="Credit Score Distribution/ Imbalance", barmode="group")
    

    fig_corr.update_layout(
    width=1200,   # width in pixels
    height=800,   # height in pixels
    title_x=0.5,  # center title
    )
    st.plotly_chart(fig_dist, use_container_width= True)

    # st.write("""
    # - Strong correlation between **Debt_to_Income_Ratio** and **Outstanding_Debt**  
    # - Customers with high **Delay_from_due_date** tend to have lower credit scores  
    # - Occupations with lower average salaries show higher default rates
    # """)

# =========================================
# 5️⃣ Feature Engineering Tab
# =========================================

    st.header("🧮 Feature Engineering")

    st.markdown("""
    Feature engineering was performed to capture **repayment capacity**.
    The following derived features were added:
    """)

    engineered_features = pd.DataFrame({
        "Feature": [
            "Debt_to_Income_Ratio",
            "EMI_to_Salary_Ratio"
        ],
        "Meaning": [
            "Total outstanding debt divided by annual income",
            "Total monthly EMI divided by in-hand salary"
        ],
        "Relation to Credit Score": [
            "High ratio → higher financial stress → lower score",
            "Higher ratio → repayment burden → lower score"
        ]
    })

    st.dataframe(engineered_features)



# =========================================
# 6️⃣ Model Comparison Tab
# =========================================
elif selected_tab == "Model Comparison":
    st.header("🤖 Model Comparison")

    
    st.subheader("Aggregated CrossValidation Results")


    st.dataframe(a)

    fig = px.bar(a, x='Model' , y=["Train Accuracy",	"Test Accuracy"],
            barmode="group", title="Agg Model Performance Comparison")
    st.plotly_chart(fig)

    st.subheader("Non-Aggregated CrossValidation Results")
    st.dataframe(na)

    fig = px.bar(na, x='Model' , y=["Train Accuracy",	"Test Accuracy"],
            barmode="group", title="Non-Agg Model Performance Comparison")
    st.plotly_chart(fig)
    # st.write("With Feature Engineering:")
    # st.dataframe(df_eng)

# =========================================
# 8️⃣ Feature Importance Tab
# =========================================
elif selected_tab == "Feature Importance":
    st.header("📈 Feature Importance (Random Forest)")


    fi_df = pd.read_csv("RF_2top_features.csv", index_col= [0])

    st.dataframe(fi_df,  use_container_width=True, height=600)
    # fi_df = fi_df.sort_values("Importance", ascending=False).head(10)

    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                title="Features By Important", color="Importance", color_continuous_scale="Viridis"
                      )
    st.plotly_chart(fig)



elif  selected_tab == "Prediction":
    st.header("Predict Your Credit Score")
    # =====================================
    # Load Saved Model and Data
    # =====================================
    best_model = joblib.load("aggregated_best_credit_score_model.pkl")
    df_customer = pd.read_csv("df_customer.csv")  # Use same dataset used for training

    st.header("🔮 Credit Score Prediction")

    # =====================================
    # Define Column Groups
    # =====================================
    categorical_cols = [
        'Payment_of_Min_Amount', 'Credit_Mix'
    ]

    numeric_cols = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt',  'Credit_History_Age',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

    # =====================================
    # Define Options for Categorical Columns
    # =====================================
    category_options = {
        'Payment_of_Min_Amount': ['No', 'Yes'],
        'Credit_Mix': ['Bad', 'Standard', 'Good']

    }

    # =====================================
    # Build Input Form
    # =====================================
    st.subheader("🧾 Input Features")

    with st.form("prediction_form"):
        st.markdown("**Categorical Features**")
        user_cats = {}
        for col in categorical_cols:
            user_cats[col] = st.selectbox(col, category_options[col])

        st.markdown("**Numerical Features**")
        user_nums = {}
        for col in numeric_cols:
            col_min = float(df_customer[col].min())
            col_max = float(df_customer[col].max())
            col_mean = float(df_customer[col].mean())
            user_nums[col] = st.slider(col, min_value=col_min, max_value=col_max, value=col_mean)

        # Ratios from user inputs
        Debt_to_Income_Ratio = user_nums['Outstanding_Debt'] / user_nums['Annual_Income']
        EMI_to_Salary_Ratio = user_nums['Total_EMI_per_month'] / user_nums['Monthly_Inhand_Salary']

        # Submit button
        submitted = st.form_submit_button("🚀 Predict Credit Score")

    # Make prediction after submit
    if submitted:
        input_data = {**user_cats, **user_nums,
                    'Debt_to_Income_Ratio': Debt_to_Income_Ratio,
                    'EMI_to_Salary_Ratio': EMI_to_Salary_Ratio}

        input_df = pd.DataFrame([input_data])
        prediction = best_model.predict(input_df)
        predicted_class = prediction[0]

        if predicted_class == 1:
             st.success("🎯 **Predicted Credit Score Class:** Poor ")
        elif predicted_class == 2:
             st.success("🎯 **Predicted Credit Score Class:** Standard ")
        else:
            st.success("🎯 **Predicted Credit Score Class:** Good ")

 


