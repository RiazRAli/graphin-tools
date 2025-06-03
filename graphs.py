import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Graphing Dashboard", layout="wide")
st.title("📊 Graphing Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded data with shape {df.shape}")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found.")
    else:
        # Step 1: Filtering
        filter_by = st.radio("Filter dataset by category?", ["No", "Yes"])
        if filter_by == "Yes" and cat_cols:
            filter_col = st.selectbox("Select a column to filter by", cat_cols)
            filter_values = st.multiselect(f"Select values from '{filter_col}'", df[filter_col].dropna().unique())
            if filter_values:
                df = df[df[filter_col].isin(filter_values)]
                st.success(f"Filtered data: {len(df)} rows")
            else:
                st.warning("No values selected; using full dataset.")

        # Step 2: Select plot type and grouping
        plot_type = st.selectbox(
            "Choose a graph type",
            ["Box Plot", "Interval Plot", "Histogram", "Curve", "Smoothed Histogram", "Dot Plot", "Matrix Plots", "Pareto Chart"]
        )


        selected_cols = st.multiselect("Select numeric columns to plot", numeric_cols, default=numeric_cols)

        group_by = None
        if cat_cols and plot_type not in ["Matrix Plots"]:
            group_by = st.selectbox("Group results by category (optional)", ["None"] + cat_cols)
            if group_by == "None":
                group_by = None

        # Step 3: Plot
        if selected_cols:
            st.subheader(f"{plot_type} for {', '.join(selected_cols)}")

            # --- Box Plot ---
            if plot_type == "Box Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                if group_by:
                    melted = df[[group_by] + selected_cols].melt(id_vars=group_by, var_name="Variable", value_name="Value")
                    sns.boxplot(data=melted, x=group_by, y="Value", hue="Variable", ax=ax)
                    ax.set_title(f"Box Plot by {group_by}")
                else:
                    sns.boxplot(data=df[selected_cols], ax=ax)
                    ax.set_title("Box Plot of Selected Columns")
               with st.container():
                st.pyplot(fig)
                st.caption("Created by: Riaz Ali, Rev 0 – May 2025")

            # --- Histogram ---
            elif plot_type == "Histogram":
                fig, ax = plt.subplots(figsize=(10, 6))
                if group_by:
                    for name, group in df.groupby(group_by):
                        for col in selected_cols:
                            sns.histplot(group[col], label=f"{col} - {name}", kde=False, stat="density", ax=ax)
                else:
                    for col in selected_cols:
                        sns.histplot(df[col], label=col, kde=False, stat="density", ax=ax)
                ax.set_title("Histogram")
                ax.legend()
                st.pyplot(fig)
                st.caption("Created by: Riaz Ali, Rev 0 – May 2025")


            # --- Curve (KDE) ---
            elif plot_type == "Curve":
                fig, ax = plt.subplots(figsize=(10, 6))
                if group_by:
                    for name, group in df.groupby(group_by):
                        for col in selected_cols:
                            sns.kdeplot(group[col].dropna(), label=f"{col} - {name}", ax=ax)
                else:
                    for col in selected_cols:
                        sns.kdeplot(df[col].dropna(), label=col, ax=ax)
                ax.set_title("Density Curve (KDE)")
                ax.legend()
                st.pyplot(fig)
                st.caption("Created by: Riaz Ali, Rev 0 – May 2025")


            # --- Smoothed Curve ---
            elif plot_type == "Smoothed Histogram":
                from scipy.stats import norm

                fig, ax = plt.subplots(figsize=(10, 6))
                x_range = st.slider("X-Range Span (in Std Devs)", 2, 6, 4)

                if group_by:
                    for name, group in df.groupby(group_by):
                        for col in selected_cols:
                            series = group[col].dropna()
                            if len(series) > 1:
                                mu, sigma = series.mean(), series.std()
                                x = np.linspace(mu - x_range*sigma, mu + x_range*sigma, 200)
                                y = norm.pdf(x, mu, sigma)
                                ax.plot(x, y, label=f"{col} - {name}")
                else:
                    for col in selected_cols:
                        series = df[col].dropna()
                        if len(series) > 1:
                            mu, sigma = series.mean(), series.std()
                            x = np.linspace(mu - x_range*sigma, mu + x_range*sigma, 200)
                            y = norm.pdf(x, mu, sigma)
                            ax.plot(x, y, label=col)

                ax.set_title("Normal Curve Fit")
                ax.set_ylabel("Probability Density")
                ax.legend()
                st.pyplot(fig)
                st.caption("Created by: Riaz Ali, Rev 0 – May 2025")




            # --- Dot Plot ---
            elif plot_type == "Dot Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if group_by:
                    # Melt the data into long-form for grouped dot plotting
                    melted = df[[group_by] + selected_cols].melt(id_vars=group_by, var_name="Variable", value_name="Value")
                    sns.stripplot(data=melted, x=group_by, y="Value", hue="Variable", dodge=True, jitter=True, ax=ax)
                    ax.set_title(f"Dot Plot by {group_by}")
                    ax.legend(title="Variable")
                else:
                    # Melt into variable-name groupings
                    melted = df[selected_cols].melt(var_name="Variable", value_name="Value")
                    sns.stripplot(data=melted, x="Variable", y="Value", jitter=True, ax=ax)
                    ax.set_title("Dot Plot of Selected Columns")

                st.pyplot(fig)
                st.caption("Created by: Riaz Ali, Rev 0 – May 2025")



            # --- Interval Plot ---
            elif plot_type == "Interval Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                if group_by:
                    categories = df[group_by].dropna().unique()
                    x = np.arange(len(categories))
                    width = 0.8 / len(selected_cols)

                    for i, col in enumerate(selected_cols):
                        means = df.groupby(group_by)[col].mean()
                        stds = df.groupby(group_by)[col].std()
                        offsets = x + i * width - (width * (len(selected_cols) - 1) / 2)
                        ax.errorbar(offsets, means, yerr=stds, fmt='o', capsize=5, label=col)

                    ax.set_xticks(x)
                    ax.set_xticklabels(categories)
                    ax.set_xlabel(group_by)
                    ax.set_ylabel("Mean Value")
                    ax.set_title(f"Interval Plot (Mean ± Std Dev) by {group_by}")
                    ax.legend(title="Variable")
                else:
                    means = df[selected_cols].mean()
                    stds = df[selected_cols].std()
                    ax.errorbar(x=selected_cols, y=means, yerr=stds, fmt='o', capsize=5)
                    ax.set_xlabel("Variable")
                    ax.set_ylabel("Mean Value")
                    ax.set_title("Interval Plot (Mean ± Std Dev)")
                st.pyplot(fig)
                st.caption("Created by: Riaz Ali, Rev 0 – May 2025")


            # --- Matrix Plots ---
            elif plot_type == "Matrix Plots":
                if len(selected_cols) < 2:
                    st.warning("Matrix plots require at least 2 numeric columns.")
                else:
                    fig = sns.pairplot(df[selected_cols])
                    st.pyplot(fig)
                    st.caption("Created by: Riaz Ali, Rev 0 – May 2025")


            # --- Pareto Chart ---
            elif plot_type == "Pareto Chart":
                for col in selected_cols:
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    if group_by:
                        data = df.groupby(group_by)[col].sum().sort_values(ascending=False)
                    else:
                        data = df[col].value_counts().sort_values(ascending=False)
                    cum_pct = data.cumsum() / data.sum()
                    data.plot(kind='bar', ax=ax1, color='skyblue')
                    ax1.set_ylabel("Frequency")
                    ax2 = ax1.twinx()
                    cum_pct.plot(ax=ax2, color='red', marker='D')
                    ax2.set_ylabel("Cumulative %")
                    ax2.axhline(0.8, color='gray', linestyle='--')
                    ax1.set_title(f"Pareto Chart for {col}" + (f" by {group_by}" if group_by else ""))
                    st.pyplot(fig)
                    st.caption("Created by: Riaz Ali, Rev 0 – May 2025")

        else:
            st.warning("Please select at least one numeric column.")
