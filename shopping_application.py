import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots
from pyexpat import model
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

st.title("Buyer Classification Made Easy")
st.write("View the Graphs / Upload your data, and this app will help classify whether a customer is a Buyer or Non-Buyer.")
st.sidebar.title("Click to Navigate")
option = st.sidebar.radio("Choose an option", ('KPI Dashboard','Data Visualization', 'Model Prediction'))

conn = sqlite3.connect('relationalDB_shopping.db')
def calculate_kpis(region, month, visitor_type):
    # Base query
    query = """
    SELECT Revenue
    FROM Metrics
    WHERE 1=1
    """

    # Add filters dynamically
    if region != 'All':
        query += f" AND Region = '{region}'"
    if month != 'All':
        query += f" AND Month = '{month}'"
    if visitor_type != 'All':
        query += f" AND VisitorType = '{visitor_type}'"

    # Fetch filtered data
    filtered_data = pd.read_sql_query(query, conn)

    # Calculate KPIs
    total_sessions = len(filtered_data)
    total_buyers = len(filtered_data[filtered_data['Revenue'] == 'Buyers'])
    conversion_rate = (total_buyers / total_sessions) * 100 if total_sessions > 0 else 0

    conversion_rate = round(conversion_rate, 2)

    return {
        "total_sessions": total_sessions,
        "total_buyers": total_buyers,
        "conversion_rate": conversion_rate
    }

def revenue_pie_chart(kpis):
    labels = ['Buyers', 'Non-Buyers']
    values = [kpis['total_buyers'], kpis['total_sessions'] - kpis['total_buyers']]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo="percent+label",
        marker=dict(colors=['#ff7f0e', '#1f77b4'])
    ))
    fig.update_layout(
        title="Revenue Distribution",
        title_x=0.5,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

def sessions_vs_buyers(kpis):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['Sessions', 'Buyers'],
        y=[kpis['total_sessions'], kpis['total_buyers']],
        name="Sessions & Buyers",
        marker=dict(color=['#ff7f0e', '#1f77b4'])
    ))

    fig.update_layout(
        title="Sessions vs Buyers",
        template='plotly_white',
        xaxis_title="Category",
        yaxis_title="Count",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def update_kpis(region, month, visitor_type):
    kpis = calculate_kpis(region, month, visitor_type)

    st.subheader("Key Performance Indicators (KPIs)")

    st.write(f"**Total Sessions:** {kpis['total_sessions']}")
    st.write(f"**Total Buyers:** {kpis['total_buyers']}")
    st.write(f"**Conversion Rate:** {kpis['conversion_rate']}%")

    revenue_pie_chart(kpis)
    sessions_vs_buyers(kpis)

# KPI Dashboard Section
if option == 'KPI Dashboard':
    # Fetch unique values for dropdown options
    data = pd.read_sql_query("SELECT DISTINCT Region, Month, VisitorType FROM Metrics", conn)
    region_options = data['Region'].dropna().unique()
    month_options = data['Month'].dropna().unique()
    visitor_type_options = data['VisitorType'].dropna().unique()

    # Create dropdown widgets using Streamlit's built-in functions
    region = st.selectbox('Region:', ['All'] + list(region_options))
    month = st.selectbox('Month:', ['All'] + list(month_options))
    visitor_type = st.selectbox('Visitor Type:', ['All'] + list(visitor_type_options))

    # Call the update_kpis function based on the selected filters
    update_kpis(region, month, visitor_type)


elif option == 'Data Visualization':
    # Establish a connection to the SQLite database
    conn = sqlite3.connect('relationalDB_shopping.db')

    # Revenue Distribution Query
    query = "SELECT Revenue, COUNT(*) AS Count FROM Metrics GROUP BY Revenue;"
    revenue_dist = pd.read_sql_query(query, conn)

    # Plot Revenue Distribution
    fig = go.Figure(data=[go.Bar(x=revenue_dist['Revenue'], y=revenue_dist['Count'], marker_color=['skyblue', 'orange'])])
    fig.update_layout(title='Revenue Distribution', xaxis_title='Revenue', yaxis_title='Count', xaxis_tickangle=0, template="plotly_white")
    st.plotly_chart(fig)

    # **Observation**:
    st.markdown("""
    **Insight**:
    - The plot shows a significant imbalance, with a larger number of non-buyers compared to buyers.
    - This suggests that the dataset might be biased towards the majority class of non-buyers, which could impact model performance.
    - Addressing this imbalance through methods like resampling or using models robust to imbalance may improve classification results.
    """)

    # Fetch other columns for distribution analysis
    query = """
    SELECT Administrative, Informational, ProductRelated, TotalDuration, Revenue, PageValues
    FROM Metrics JOIN Analytics ON Metrics.AnalyticsID = Analytics.AnalyticsID;
    """
    data = pd.read_sql_query(query, conn)

    # Histograms for Columns
    fig = make_subplots(rows=5, cols=1, subplot_titles=["Administrative", "Informational", "ProductRelated (Log Scale)", "TotalDuration", "PageValues"])
    columns = ['Administrative', 'Informational', 'ProductRelated', 'TotalDuration', 'PageValues']
    for i, col in enumerate(columns):
        if col == 'ProductRelated':
            fig.add_trace(go.Histogram(x=data[col], name=col), row=i+1, col=1)
            fig.update_yaxes(type='log', row=i+1, col=1)
        elif col == 'TotalDuration':
            fig.add_trace(go.Histogram(x=data[col], name=col), row=i+1, col=1)
            fig.update_xaxes(range=[0, 30000], row=i+1, col=1)
        elif col == 'PageValues':
            fig.add_trace(go.Histogram(x=data[col], name=col), row=i+1, col=1)
            fig.update_xaxes(range=[0, 50], row=i+1, col=1)
        else:
            fig.add_trace(go.Histogram(x=data[col], name=col), row=i+1, col=1)
    fig.update_layout(title="Histograms with Scale Adjustments", height=1200, showlegend=False)
    st.plotly_chart(fig)

    # **Observations for Histograms**:
    st.markdown("""
    **Insights**:
    - **Administrative & Informational**: Both distributions are highly right-skewed, indicating that most users visit only a few administrative and informational pages.
    - **ProductRelated**: There is a long tail suggesting some users explore many product-related pages. This could indicate higher engagement and a potential higher likelihood of making a purchase.
    - **TotalDuration**: The duration distribution is also right-skewed, with a small number of users spending disproportionately long periods on the site. These users may be highly engaged and more likely to convert.
    - **PageValues**: The majority of pages have low estimated values, but some pages with significantly higher values may be contributing more to revenue.
    """)

    # Visitor Type vs Revenue Distribution
    query = "SELECT VisitorType, Revenue, COUNT(*) AS Count FROM Metrics GROUP BY VisitorType, Revenue;"
    visitor_type = pd.read_sql_query(query, conn)
    data_by_revenue = visitor_type.pivot(index='VisitorType', columns='Revenue', values='Count').fillna(0)
    fig = go.Figure()
    for revenue_category in data_by_revenue.columns:
        fig.add_trace(go.Bar(x=data_by_revenue.index, y=data_by_revenue[revenue_category], name=f'Revenue: {revenue_category}'))
    fig.update_layout(barmode='stack', title='Visitor Type Distribution by Revenue', xaxis_title='Visitor Type', yaxis_title='Count', legend_title='Revenue', xaxis=dict(tickangle=45), template='plotly_white')
    st.plotly_chart(fig)

    # **Observation**:
    st.markdown("""
    **Insight**:
    - Returning visitors are the most likely to either make a purchase or not, with a higher count of both buyers and non-buyers. This suggests that return visitors tend to engage more with the site.
    - Returning users also have a higher proportion of buyers compared to non-buyers, making them a key audience for conversion.
    """)

    # Weekend Impact on Revenue
    query = "SELECT Weekend, Revenue, COUNT(*) AS Count FROM Metrics GROUP BY Weekend, Revenue;"
    weekend_impact = pd.read_sql_query(query, conn)
    weekend_pivot = weekend_impact.pivot(index='Weekend', columns='Revenue', values='Count').fillna(0)
    fig = go.Figure()
    for revenue_category in weekend_pivot.columns:
        fig.add_trace(go.Bar(x=weekend_pivot.index, y=weekend_pivot[revenue_category], name=f'Revenue: {revenue_category}', marker_color='skyblue' if revenue_category == 'Non-Buyers' else 'orange'))
    fig.update_layout(barmode='group', title='Weekend Impact on Revenue', xaxis_title='Weekend', yaxis_title='Count', legend_title='Revenue', xaxis=dict(tickangle=0), template='plotly_white', legend=dict(title='Revenue', x=1.05, y=1))
    st.plotly_chart(fig)

    # **Observation**:
    st.markdown("""
    **Insight**:
    - Weekends see higher visitor counts and more conversions, with a slightly higher proportion of buyers than on weekdays. This suggests that weekends are a more favorable time for attracting buyers.
    """)

    # Revenue Distribution by Month
    query = "SELECT Month, Revenue, COUNT(*) AS Count FROM Metrics GROUP BY Month, Revenue ORDER BY Month;"
    df = pd.read_sql_query(query, conn)
    fig = px.bar(df, x='Month', y='Count', color='Revenue', title="Revenue Distribution by Month", labels={"Month": "Month", "Count": "Count of Revenue (0s and 1s)"}, color_discrete_map={0: 'blue', 1: 'orange'}, barmode='stack')
    st.plotly_chart(fig)


    # **Observation**:
    st.markdown("""
    **Insight**:
    - May and November have the highest number of both buyers and non-buyers, indicating that these months are the most active for website traffic and conversions.
    The proportion of buyers to non-buyers is highest in Nov, maybe because of Black Friday and Thanksgiving offers, suggesting that Nov is the most favorable month for driving sales and increasing revenue.
    Months like February and July have a relatively lower number of both buyers and non-buyers, indicating lower activity during these periods.
    """)

    # Scatterplots
    query = """
    SELECT Administrative, Informational, ProductRelated, TotalDuration, Revenue
    FROM Metrics JOIN Analytics ON Metrics.AnalyticsID = Analytics.AnalyticsID;
    """
    df = pd.read_sql_query(query, conn)
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Administrative vs Revenue", "Informational vs Revenue", "ProductRelated vs Revenue", "TotalDuration vs Revenue"))
    fig.add_trace(go.Scatter(x=df['Administrative'], y=df['Revenue'], mode='markers', name="Administrative"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Informational'], y=df['Revenue'], mode='markers', name="Informational"), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['ProductRelated'], y=df['Revenue'], mode='markers', name="ProductRelated"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['TotalDuration'], y=df['Revenue'], mode='markers', name="TotalDuration"), row=2, col=2)
    fig.update_layout(title="Scatter Plot",xaxis_title="Administrative", xaxis2_title="Informational", xaxis3_title="ProductRelated", xaxis4_title="TotalDuration", yaxis_title="Revenue", yaxis3_title="Revenue", height=700, width=900)
    st.plotly_chart(fig)

    # **Observation**:
    st.markdown("""
    **Insight**:
    - **Administrative & Informational**: These variables show weak correlation with revenue, with most users visiting only a few pages in these categories.
    - **ProductRelated**: There is a slight positive correlation, indicating that users exploring more product-related pages are more likely to purchase.
    - **TotalDuration**: There is a weak positive correlation, where longer sessions slightly increase the likelihood of purchases.
    """)

    # Correlation Heatmap
    query = """
    SELECT Administrative, Informational, ProductRelated, TotalDuration, SpecialDay, BounceRates, ExitRates, PageValues
    FROM Metrics JOIN Analytics ON Metrics.AnalyticsID = Analytics.AnalyticsID;
    """
    df = pd.read_sql_query(query, conn)
    corr_matrix = df.corr().round(2)
    fig = ff.create_annotated_heatmap(z=corr_matrix.values, x=corr_matrix.columns.tolist(), y=corr_matrix.columns.tolist(), colorscale='Viridis', showscale=True)
    fig.update_layout(title="Correlation Heatmap")
    st.plotly_chart(fig)

    st.markdown("""
    **Insight**:
    - **Strong Positive Correlations**:
      **ExitRates** and **BounceRates** have a strong positive correlation (0.91), suggesting that pages with high exit rates often have high bounce rates as well.

    - **Moderate Positive Correlations**:
      **Administrative, Informational, and ProductRelated** variables show moderate positive correlations with each other, suggesting that users who engage with one type of content tend to engage with others as well.

    - **Weak Negative Correlations**:
     **PageValues** has a weak negative correlation with **ExitRates and BounceRates**, indicating that pages with higher values tend to have lower exit and bounce rates.
    """)

    # Close the database connection
    conn.close()

elif option == 'Model Prediction':
    # Data Upload Section
    st.subheader("1. Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Remove unnecessary columns
        columns_to_exclude = ["Customer Name", "Email", "City", "State"]
        data = data.drop(columns=[col for col in columns_to_exclude if col in data.columns], errors='ignore')

        st.write("### Dataset Preview")
        st.write(data.head())

        # Step 1: Plot the Correlation Matrix
        st.subheader("2. Correlation Matrix and Multicollinearity Check")

        # Separate features and target
        X = data.select_dtypes(include=['number'])  # Select numeric columns only for correlation matrix
        corr_matrix = X.corr()

        # Visualize the correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        st.pyplot(fig)

        # Step 2: Identify and Remove Multicollinear Features based on correlation > 0.9
        corr_threshold = 0.9
        drop_columns = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                    colname = corr_matrix.columns[i]
                    drop_columns.add(colname)

        # Remove the correlated columns, but do not display this information in the app
        X = X.drop(columns=drop_columns)

        # Step 3: Check Variance Inflation Factor (VIF) for multicollinearity
        st.subheader("3. VIF Analysis")
        X_with_const = add_constant(X)  # Add constant for VIF calculation
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
        st.write("### VIF Values")
        st.write(vif_data)

        # Remove columns with VIF > 10
        high_vif_columns = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
        if "const" in high_vif_columns:
            high_vif_columns.remove("const")

        # Remove the high VIF columns, but do not display this information in the app
        X = X.drop(columns=high_vif_columns)

        # Step 4: Allow the user to select the target and feature columns
        st.subheader("4. Configure the Classification Model")

        columns = data.columns
        # Allow the user to select the target column (default to 'Revenue' if available)
        target_column = st.selectbox("Select the target column (e.g., Revenue, Buyer/Non-Buyer):", options=columns, index=columns.get_loc('Revenue') if 'Revenue' in columns else 0)

        # Allow the user to select the feature columns from the remaining ones
        feature_columns = st.multiselect("Select the feature columns:", options=X.columns)

        if not feature_columns:
            st.warning("Please select at least one feature column.")
        else:
            # Preprocess data for model
            X_final = data[feature_columns]
            y = data[target_column]

            # Handle non-numeric features, excluding "Month"
            if "Month" in X_final.columns:
                st.info("'Month' will be treated as a categorical variable without one-hot encoding.")
                X_final["Month"] = X_final["Month"].astype('category').cat.codes

            categorical_features = [col for col in X_final.columns if X_final[col].dtype == 'object' and col != "Month"]
            X_final = pd.get_dummies(X_final, columns=categorical_features, drop_first=True)

            # Encode non-numeric target if necessary
            if y.dtype == 'object' or y.dtype.name == 'category':
                st.info("The target column contains non-numeric values. They will be encoded.")
                y = y.astype('category').cat.codes

            # Step 5: Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

            # Train model with default settings
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.write("### Model Performance")
            st.write(f"Accuracy: {accuracy:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Feature Importance
            # Feature Importance using Plotly
            feature_importance = pd.DataFrame({"Feature": X_final.columns, "Importance": model.feature_importances_})
            feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

            st.write("### Feature Importance")
            fig = go.Figure([go.Bar(
                x=feature_importance['Importance'],
                y=feature_importance['Feature'],
                orientation='h',
                marker=dict(color='blue')
            )])

            fig.update_layout(title="Feature Importance", xaxis_title="Importance", yaxis_title="Feature", template="plotly_white")
            st.plotly_chart(fig)

           # Feature Impact using Plotly
            st.write("### Feature Impact Analysis")
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            perm_importance_df = pd.DataFrame({
                "Feature": X_final.columns,
                "Importance": perm_importance.importances_mean
            }).sort_values(by="Importance", ascending=False)

            fig = go.Figure([go.Bar(
                x=perm_importance_df['Importance'],
                y=perm_importance_df['Feature'],
                orientation='h',
                marker=dict(color='green')
            )])

            fig.update_layout(title="Feature Impact", xaxis_title="Importance", yaxis_title="Feature", template="plotly_white")
            st.plotly_chart(fig)


            # Step 6: Prediction interface
            st.subheader("6. Make a Prediction")
            st.write("Enter customer data to classify whether they are a Buyer or Non-Buyer.")

            input_data = {}

            # Collect input data
            for feature in X_final.columns:
                if feature == "Month":
                    months = data["Month"].unique()
                    selected_month = st.selectbox("Select Month", options=months)
                    input_data[feature] = pd.Series(selected_month).astype('category').cat.codes[0]
                else:
                    min_value = X_final[feature].min()
                    max_value = X_final[feature].max()
                    value = st.slider(f"{feature}", min_value=float(min_value), max_value=float(max_value), value=float(min_value))
                    input_data[feature] = value

            # Validation: Check if all input values are zero or invalid
            if st.button("Classify Customer"):
                if all(value == 0 for value in input_data.values()):
                    st.warning("Please provide meaningful input. All values are zero, which is not valid for classification.")
                else:
                    input_df = pd.DataFrame([input_data])
                    for feature in X_final.columns:
                        if feature != "Month" and X_final[feature].dtype == 'object':
                            input_df[feature] = pd.get_dummies(input_df[feature], drop_first=True).iloc[0]

                    prediction = model.predict(input_df)[0]
                    result = "Buyer" if prediction == 1 else "Non-Buyer"
                    st.write(f"### Customer Classification: {result}")

    else:
        st.info("Please upload a CSV file to proceed.")
