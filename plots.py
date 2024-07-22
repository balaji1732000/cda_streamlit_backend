import nltk
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from nltk import regexp_tokenize


# Single attribute visualization
def distribution_histogram(df, attribute):
    """
    Interactive histogram of the distribution of a single attribute using Plotly.
    """
    if df[attribute].dtype == "object" or pd.api.types.is_categorical_dtype(
        df[attribute]
    ):
        codes, uniques = pd.factorize(df[attribute])
        temp_df = pd.DataFrame({attribute: codes})
        fig = px.histogram(temp_df, x=attribute, color_discrete_sequence=["#e17160"])
        fig.update_xaxes(
            tickvals=list(range(len(uniques))), ticktext=uniques, tickangle=45
        )
    else:
        fig = px.histogram(df, x=attribute, color_discrete_sequence=["#e17160"])

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        autosize=True,  # Ensure the plot resizes to fit within the container
    )

    return fig


def bar_chart(df, attribute):
    """
    Interactive bar chart of the distribution of a single attribute using Plotly.
    """
    if attribute in df.columns:
        value_counts = df[attribute].value_counts()
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Bar Chart of {attribute}",
            labels={"x": attribute, "y": "Count"},
            color=value_counts.index,
            color_discrete_sequence=px.colors.sequential.Cividis_r,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black"),
            yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        return fig


def distribution_boxplot(df, attribute):
    """
    Interactive boxplot of the distribution of a single attribute using Plotly.
    """
    if df[attribute].dtype == "object" or pd.api.types.is_categorical_dtype(
        df[attribute]
    ):
        return -1
    fig = px.box(df, y=attribute, color_discrete_sequence=["#32936f"])
    fig.update_layout(
        title=f"Boxplot of {attribute}",
        plot_bgcolor="#f0f0f0",
        paper_bgcolor="#f0f0f0",
        yaxis_title=attribute,
        font=dict(color="black"),
    )
    return fig


def count_Y(df, Y_name):
    """
    Interactive donut chart of the distribution of a single attribute using Plotly.
    """
    if Y_name in df.columns and df[Y_name].nunique() >= 1:
        value_counts = df[Y_name].value_counts()
        fig = px.pie(
            names=value_counts.index,
            values=value_counts.values,
            title=f"Distribution of {Y_name}",
            hole=0.5,
        )
        fig.update_layout(
            legend=dict(
                yanchor="auto",
                y=0.5,
                xanchor="auto",
                x=1.2,  # Adjust this value to move the legend further to the right
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
            paper_bgcolor="rgba(0, 0, 0, 0)",
            autosize=True,  # Ensure the plot resizes to fit within the container
        )
        return fig


def density_plot(df, column_name):
    """
    Interactive density plot of the distribution of a single attribute using Plotly.
    """
    if column_name in df.columns:
        fig = px.density_contour(
            df,
            x=column_name,
            y=column_name,
            title=f"Density Plot of {column_name}",
            color_discrete_sequence=px.colors.sequential.Inferno,
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
            paper_bgcolor="rgba(0, 0, 0, 0)",
            autosize=True,  # Ensure the plot resizes to fit within the container
        )
        return fig


def pie_chart(df, attribute):
    """
    Interactive pie chart of the distribution of a single attribute using Plotly.
    """
    if attribute in df.columns:
        value_counts = df[attribute].value_counts()
        fig = px.pie(
            names=value_counts.index,
            values=value_counts.values,
            title=f"Pie Chart of {attribute}",
            color=value_counts.index,
            color_discrete_sequence=px.colors.sequential.Cividis_r,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black"),
            yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        return fig


# Multiple attribute visualization
def box_plot(df, column_names):
    """
    Interactive box plot of multiple attributes using Plotly.
    """
    if len(column_names) > 1 and not all(
        df[column_names].dtypes.apply(lambda x: np.issubdtype(x, np.number))
    ):
        return -1
    valid_columns = [col for col in column_names if col in df.columns]
    if valid_columns:
        fig = px.box(
            df,
            y=valid_columns,
            title=f'Box Plot of {", ".join(valid_columns)}',
            color_discrete_sequence=px.colors.sequential.Cividis_r,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
            paper_bgcolor="rgba(0, 0, 0, 0)",
            autosize=True,  # Ensure the plot resizes to fit within the container
        )

        return fig


def violin_plot(df, column_names):
    """
    Interactive violin plot of multiple attributes using Plotly.
    """
    if len(column_names) > 1 and not all(
        df[column_names].dtypes.apply(lambda x: np.issubdtype(x, np.number))
    ):
        return -1
    valid_columns = [col for col in column_names if col in df.columns]
    if valid_columns:
        fig = px.violin(
            df,
            y=valid_columns,
            title=f'Violin Plot of {", ".join(valid_columns)}',
            color_discrete_sequence=px.colors.sequential.Cividis_r,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        return fig


def strip_plot(df, column_names):
    """
    Interactive strip plot of multiple attributes using Plotly.
    """
    if len(column_names) > 1 and not all(
        df[column_names].dtypes.apply(lambda x: np.issubdtype(x, np.number))
    ):
        return -1
    valid_columns = [col for col in column_names if col in df.columns]
    if valid_columns:
        fig = px.strip(
            df,
            y=valid_columns,
            title=f'Strip Plot of {", ".join(valid_columns)}',
            color_discrete_sequence=px.colors.sequential.Cividis_r,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
            paper_bgcolor="rgba(0, 0, 0, 0)",
            autosize=True,  # Ensure the plot resizes to fit within the container
        )
        return fig


def scatter_matrix(df, attributes):
    """
    Interactive scatter matrix of multiple attributes using Plotly.
    """
    if all(attr in df.columns for attr in attributes):
        fig = px.scatter_matrix(df[attributes])
        fig.update_layout(
            title="Scatter Matrix",
            width=1000,
            height=1000,
        )
        return fig


def multi_plot_scatter(df, selected_attributes):
    """
    Interactive scatter plot of multiple attributes using Plotly.
    """
    if len(selected_attributes) < 2:
        return -1
    fig = px.scatter(
        df,
        x=selected_attributes[0],
        y=selected_attributes[1],
        color=selected_attributes[1],
        title=f"Scatter Plot of {selected_attributes[0]} vs {selected_attributes[1]}",
    )
    return fig


def multi_plot_line(df, selected_attributes):
    """
    Interactive line plot of multiple attributes using Plotly.
    """
    if not all(
        df[selected_attributes].dtypes.apply(lambda x: np.issubdtype(x, np.number))
    ):
        return -1
    if len(selected_attributes) >= 2:
        fig = px.line(
            df,
            x=df.index,
            y=selected_attributes,
            title=f"Line Plot of {selected_attributes[0]} vs {selected_attributes[1]}",
            labels={
                selected_attributes[0]: selected_attributes[0],
                "value": selected_attributes[1],
            },
        )
        return fig
    else:
        return -2


def multi_plot_heatmap(df, selected_attributes):
    """
    Interactive correlation heatmap of multiple attributes using Plotly.
    """
    if not all(
        df[selected_attributes].dtypes.apply(lambda x: np.issubdtype(x, np.number))
    ):
        return -1

    if len(selected_attributes) >= 1:
        corr_matrix = df[selected_attributes].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Heatmap of Correlation",
            color_continuous_scale=px.colors.sequential.Plasma,  # Adjust color scale here
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
            autosize=True,  # Ensure the plot resizes to fit within the container
        )
        return fig


# Overall visualization
@st.cache_data
def correlation_matrix(df):
    """
    Interactive correlation heatmap of all attributes using Plotly.
    """
    corr_matrix = df.corr()
    fig = px.imshow(
        corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix"
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        autosize=True,  # Ensure the plot resizes to fit within the container
    )
    return fig


@st.cache_data
def correlation_matrix_plotly(df):
    """
    Interactive correlation heatmap of all attributes using Plotly.
    """
    corr_matrix = df.corr()
    labels = corr_matrix.columns
    text = [
        [f"{corr_matrix.iloc[i, j]:.2f}" for j in range(len(labels))]
        for i in range(len(labels))
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=labels,
            y=labels,
            colorscale="Viridis",
            colorbar=dict(title="Correlation"),
            text=text,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Correlation Matrix Between Attributes",
        xaxis=dict(tickmode="linear"),
        yaxis=dict(tickmode="linear"),
        width=800,
        height=700,
    )
    fig.update_layout(font=dict(size=10))
    return fig


@st.cache_data
def list_all(df, max_plots=16, max_categories=10):
    """
    Display interactive histograms of all attributes in the DataFrame.
    Parameters:
    - df: DataFrame containing the data.
    - max_plots: Maximum number of plots to display.
    - max_categories: Maximum number of categories to show for categorical variables.
    """
    # Calculate the number of plots to display (up to 16)
    num_plots = min(len(df.columns), max_plots)
    nrows = int(np.ceil(num_plots / 4))
    ncols = min(num_plots, 4)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=df.columns[:num_plots])

    for i, column in enumerate(df.columns[:num_plots]):
        row = i // ncols + 1
        col = i % ncols + 1
        if df[column].dtype == "object" or df[column].nunique() < max_categories:
            # Categorical variable
            value_counts = df[column].value_counts()
            if len(value_counts) > max_categories:
                value_counts = value_counts[:max_categories]
                value_counts["Other"] = df[column].isin(value_counts.index).sum()
            fig.add_trace(
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker=dict(color="#1867ac"),
                ),
                row=row,
                col=col,
            )
        else:
            # Numerical variable
            fig.add_trace(
                go.Histogram(x=df[column], marker=dict(color="#1867ac")),
                row=row,
                col=col,
            )

    fig.update_layout(
        height=400 * nrows,
        title_text="Attribute Distributions",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        autosize=True,  # Ensure the plot resizes to fit within the container
    )
    return fig


# Model evaluation
def confusion_metrix(model_name, model, X_test, Y_test):
    """
    Interactive confusion matrix plot for classification models using Plotly.
    """
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test, Y_pred)
    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        title=f"Confusion Matrix for {model_name}",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        autosize=True,  # Ensure the plot resizes to fit within the container
    )
    return fig


def roc(model_name, fpr, tpr):
    """
    Interactive ROC curve for classification models using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=model_name))
    fig.update_layout(
        title=f"ROC Curve - {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        font=dict(size=14),
    )
    return fig


def plot_clusters(X, labels):
    """
    Interactive scatter plot of clusters for clustering models using Plotly.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=labels,
        title="Cluster Scatter Plot",
        labels={"x": "Principal Component 1", "y": "Principal Component 2"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        autosize=True,  # Ensure the plot resizes to fit within the container
    )
    return fig


def plot_residuals(y_pred, Y_test):
    """
    Interactive residual plot for regression models using Plotly.
    """
    residuals = Y_test - y_pred
    fig = px.scatter(x=y_pred, y=residuals, trendline="lowess", title="Residual Plot")
    fig.update_layout(
        xaxis_title="Predicted Values", yaxis_title="Residuals", font=dict(size=14)
    )
    return fig


def plot_predictions_vs_actual(y_pred, Y_test):
    """
    Interactive scatter plot of predicted vs. actual values for regression models using Plotly.
    """
    fig = px.scatter(
        x=Y_test,
        y=y_pred,
        title="Actual vs. Predicted",
        labels={"x": "Actual", "y": "Predicted"},
    )
    fig.add_trace(
        go.Scatter(
            x=[Y_test.min(), Y_test.max()],
            y=[Y_test.min(), Y_test.max()],
            mode="lines",
            line=dict(dash="dash"),
        )
    )
    return fig


def plot_qq_plot(y_pred, Y_test):
    """
    Interactive Quantile-Quantile plot for regression models using Plotly.
    """
    residuals = Y_test - y_pred
    fig = go.Figure()
    osm, osr = stats.probplot(residuals, dist="norm")[:2]
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Data Points"))
    fig.add_trace(go.Scatter(x=osm, y=osm, mode="lines", name="Fit Line"))
    fig.update_layout(
        title="Quantile-Quantile Plot",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Ordered Values",
        font=dict(size=14),
    )
    return fig


# Advanced Visualization
@st.cache_data
def word_cloud_plot(text):
    """
    Generates and displays a word cloud from the given text.
    The word cloud visualizes the frequency of occurrence of words in the text, with the size of each word indicating its frequency.
    :param text: The input text from which to generate the word cloud.
    :return: A matplotlib figure object containing the word cloud if successful, -1 otherwise.
    """
    try:
        words = regexp_tokenize(text, pattern="\w+")
        text_dist = nltk.FreqDist([w for w in words])
        wordcloud = WordCloud(
            width=1200, height=600, background_color="white"
        ).generate_from_frequencies(text_dist)
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        return fig
    except:
        return -1


@st.cache_data
def world_map(df, country_column, key_attribute):
    """
    Creates an interactive choropleth world map visualization based on the specified DataFrame using Plotly.
    The function highlights countries based on a key attribute, providing an interactive map that can be used to analyze geographical data distributions.
    :param df: DataFrame containing the data to be visualized.
    :param country_column: Name of the column in df that contains country names.
    :param key_attribute: Name of the column in df that contains the data to visualize on the map.
    :return: A Plotly figure object representing the choropleth map if successful, -1 otherwise.
    """
    try:
        hover_data_columns = [col for col in df.columns if col != country_column]
        fig = px.choropleth(
            df,
            locations="iso_alpha",
            color=key_attribute,
            hover_name=country_column,
            hover_data=hover_data_columns,
            color_continuous_scale=px.colors.sequential.Cividis,
            projection="equirectangular",
        )
        return fig
    except:
        return -1


def scatter_3d(df, x, y, z):
    """
    Generates an interactive 3D scatter plot from the given DataFrame using Plotly.
    Each point in the plot corresponds to a row in the DataFrame, with its position determined by three specified columns.
    Points are colored based on the values of the z-axis.
    :param df: DataFrame containing the data to be visualized.
    :param x: Name of the column in df to use for the x-axis values.
    :param y: Name of the column in df to use for the y-axis values.
    :param z: Name of the column in df to use for the z-axis values and color coding.
    :return: A Plotly figure object containing the 3D scatter plot if successful, -1 otherwise.
    """
    try:
        fig = px.scatter_3d(
            df,
            x=x,
            y=y,
            z=z,
            color=z,
            color_continuous_scale=px.colors.sequential.Plasma,  # Change color scale here
            title="3D Scatter Plot",
            labels={x: x, y: y, z: z},
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black"),
            yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        return fig
    except Exception as e:
        print(f"Error generating 3D scatter plot: {e}")
        return -1


def line_plot(df, date_column, value_column):
    """
    Interactive line plot for time series data using Plotly.
    """
    fig = px.line(
        df,
        x=date_column,
        y=value_column,
        title=f"Line Plot of {value_column} over time",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    return fig


def area_plot(df, date_column, value_column):
    """
    Interactive area plot for cumulative time series data using Plotly.
    """
    fig = px.area(
        df,
        x=date_column,
        y=value_column,
        title=f"Area Plot of {value_column} over time",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    return fig


def lag_plot(df, value_column, lag=1):
    """
    Lag plot to visualize the relationship between a value and a lagged version of itself using Plotly.
    """
    df["lagged"] = df[value_column].shift(lag)
    fig = px.scatter(
        df,
        x=value_column,
        y="lagged",
        title=f"Lag Plot of {value_column} with lag={lag}",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    return fig


# Multi-type Data Visualizations


# def pair_plot(df, attributes):
#     """
#     Interactive pair plot to visualize relationships between pairs of variables using Seaborn and Plotly.
#     """
#     sns_plot = sns.pairplot(df[attributes])
#     fig = plt.figure()
#     sns_plot.savefig(fig, format="png")
#     return fig


def radar_chart(df, attributes, group_by):
    """
    Interactive radar chart to visualize multi-attribute data using Plotly.
    """
    if group_by not in df.columns:
        st.warning(f"Group by column '{group_by}' does not exist in the DataFrame.")
        return -1

    categories = attributes
    fig = go.Figure()

    for group in df[group_by].unique():
        subset = df[df[group_by] == group]
        values = subset[attributes].mean().values.flatten().tolist()
        values += values[:1]  # Ensure the radar chart is closed
        fig.add_trace(
            go.Scatterpolar(r=values, theta=categories, fill="toself", name=group)
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(df[attributes].max().max(), 1)])
        ),
        showlegend=True,
        title="Radar Chart",
    )
    return fig


# Additional functions for existing visualizations...


# Overall Visualization
@st.cache_data
def list_all(df, max_plots=16, max_categories=10):
    """
    Display interactive histograms of all attributes in the DataFrame.
    Parameters:
    - df: DataFrame containing the data.
    - max_plots: Maximum number of plots to display.
    - max_categories: Maximum number of categories to show for categorical variables.
    """
    num_plots = min(len(df.columns), max_plots)
    nrows = int(np.ceil(num_plots / 4))
    ncols = min(num_plots, 4)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=df.columns[:num_plots])

    for i, column in enumerate(df.columns[:num_plots]):
        row = i // ncols + 1
        col = i % ncols + 1
        if df[column].dtype == "object" or df[column].nunique() < max_categories:
            value_counts = df[column].value_counts()
            if len(value_counts) > max_categories:
                value_counts = value_counts[:max_categories]
                value_counts["Other"] = df[column].isin(value_counts.index).sum()
            fig.add_trace(
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker=dict(color="#1867ac"),
                ),
                row=row,
                col=col,
            )
        else:
            fig.add_trace(
                go.Histogram(x=df[column], marker=dict(color="#1867ac")),
                row=row,
                col=col,
            )

    fig.update_layout(
        height=400 * nrows,
        title_text="Attribute Distributions",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        autosize=True,
    )
    return fig


def line_plot(df, date_column, value_column):
    """
    Interactive line plot for time series data using Plotly.
    """
    fig = px.line(
        df,
        x=date_column,
        y=value_column,
        title=f"Line Plot of {value_column} over time",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    return fig


def area_plot(df, date_column, value_column):
    """
    Interactive area plot for cumulative time series data using Plotly.
    """
    fig = px.area(
        df,
        x=date_column,
        y=value_column,
        title=f"Area Plot of {value_column} over time",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    return fig


def lag_plot(df, value_column, lag=1):
    """
    Lag plot to visualize the relationship between a value and a lagged version of itself using Plotly.
    """
    df["lagged"] = df[value_column].shift(lag)
    fig = px.scatter(
        df,
        x=value_column,
        y="lagged",
        title=f"Lag Plot of {value_column} with lag={lag}",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        yaxis=dict(showgrid=True, gridcolor="#cecdcd"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    return fig


# Multi-type Data Visualizations


def pair_plot(df, attributes):
    """
    Interactive pair plot to visualize relationships between pairs of variables using Seaborn and Plotly. 
    """
    # Check if attributes exist and are numeric
    valid_attributes = [
        attr
        for attr in attributes
        if attr in df.columns and pd.api.types.is_numeric_dtype(df[attr])
    ]

    if not valid_attributes:
        st.warning("No valid attributes found for pair plot.")
        return -1

    sns_plot = sns.pairplot(df[valid_attributes])
    fig = sns_plot.fig
    return fig
