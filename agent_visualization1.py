import streamlit as st
import pandas as pd
import os
import re
import json
from local_components import card_container
from utils import developer_info_static
from plots import (
    list_all,
    distribution_histogram,
    distribution_boxplot,
    count_Y,
    box_plot,
    violin_plot,
    strip_plot,
    density_plot,
    multi_plot_heatmap,
    multi_plot_scatter,
    multi_plot_line,
    word_cloud_plot,
    world_map,
    scatter_3d,
    bar_chart,
    pie_chart,
    scatter_matrix,
    line_plot,
    area_plot,
    lag_plot,
    pair_plot,
    radar_chart,
)
import pandas.api.types as ptypes
from pygwalker.api.streamlit import StreamlitRenderer
from ai_visual_agent1 import ai_visual_agent


# Function to identify unique ID columns
def is_unique_id(df, column, threshold=0.9):
    return df[column].nunique() / len(df) > threshold


def init_session_state():
    if "selected_functionality" not in st.session_state:
        st.session_state["selected_functionality"] = None
    if "visualization_recommendations" not in st.session_state:
        st.session_state["visualization_recommendations"] = None


st.markdown(
    """  
      <style>  
        .MainMenu, footer, header, .css-1oe5cao, .css-1v3fvcr, .css-1n543e5 {visibility: hidden;}  
      .css-18e3th9 {  
            padding: 0 1rem;  
        }  
        .css-1lcbmhc, .css-1d391kg {  
            padding: 0 1rem;  
            max-width: 100% !important;  
        }  
       .selectbox-label {  
            color: black;  
            font-weight: bold;  
        }  
      .main .block-container {  
            padding: 0 1rem;  
            max-width: 100%;  
            background-color: #f0f0f0; /* Change this to your desired background color */  
        }  
               [data-testid="stStatusWidget"] {  
            visibility: hidden;  
        }  
        .main .block-container {  
            background-color: #304666; /* White background for the container */  
            border-radius: 10px;  
            padding: 2rem;  
        }  
        .stTabs [data-baseweb="tab"]:hover {  
            background-color: #f0f0f0;  
            color: #000;  
        }  
        .stTabs [data-baseweb="tab"][aria-selected="true"]{  
            background-color: #fff;  
            color: #000;  
            border-bottom: 2px solid #ff4b4b;  
        }  
        .css-1d391kg {  
            font-family: 'Arial', sans-serif;  
            font-size: 1rem;  
        }  
        .css-1hb8ztp {  
            font-size: 1rem;  
            color: #333;  
        }  
        .css-10trblm {  
            padding: 0.5rem;  
        }  
        .title {  
            font-size: 3rem; /* Font size for the title */  
            font-weight: bold;  
            color: #002B50; /* Text color */  
            text-align: center; /* Center align the text */  
            font-family: 'Arial', sans-serif; /* Font family */  
            background-color: #F0F8FF; /* Light background color */  
            padding: 1rem; /* Padding around the text */  
            border-radius: 10px; /* Rounded corners */  
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */  
            margin-bottom: 2rem; /* Space below the title */  
        }  
        .black-subheader h3 {  
            color: white;  
            text-align: center;  
        }  
        .black-subheader h4 {  
            color: white;  
            text-align: center;  
        }  
        div[data-testid="stSelectbox"] p {  
            color: white;  
        }  
      </style>  
    """,
    unsafe_allow_html=True,
)


def is_categorical(attribute, df):
    return (
        isinstance(df[attribute].dtype, pd.CategoricalDtype)
        or df[attribute].dtype == "object"
    )


def is_numeric(attribute, df):
    return ptypes.is_numeric_dtype(df[attribute])


def is_text(attribute, df):
    return df[attribute].dtype == "object" and not isinstance(
        df[attribute].dtype, pd.CategoricalDtype
    )


def detect_datetime_column(df):
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return column
        try:
            if pd.to_datetime(df[column], errors="coerce").notna().all():
                return column
        except:
            continue
    return None


def display_word_cloud(text):
    _, word_cloud_col, _ = st.columns([1, 3, 1])
    with word_cloud_col:
        word_fig = word_cloud_plot(text)
        if word_fig == -1:
            st.error("Data not supported")
        else:
            st.pyplot(word_fig)


def load_visualization_recommendations(file_path="visualization_recommendations.txt"):
    with open(file_path, "r") as file:
        file_content = file.read()
    json_match = re.search(r"```json(.*?)```", file_content, re.DOTALL)
    if json_match:
        json_content = json_match.group(1).strip()
        print("JSON content extracted successfully.")
    else:
        print("No JSON content found.")
        return None

    try:
        recommendations = json.loads(json_content)
        print("JSON content parsed successfully.")
        print(
            f"Recommendations structure: {json.dumps(recommendations, indent=4)}"
        )  # Debug print
        return recommendations
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON content: {e}")
        return None


def get_default_recommendation(recommendations, attribute_type, chart_type):
    try:
        if attribute_type in recommendations:
            if chart_type in recommendations[attribute_type]:
                attributes = recommendations[attribute_type][chart_type].get(
                    "Attributes", []
                )
                if attributes and isinstance(attributes, list):
                    return attributes
                else:
                    st.warning(
                        f"No attributes found for {chart_type} in {attribute_type}. Justification: {recommendations[attribute_type][chart_type].get('Justification', 'No reason provided.')}"
                    )
            else:
                st.warning(f"Chart type {chart_type} not found in {attribute_type}.")
        else:
            st.warning(f"Attribute type {attribute_type} not found in recommendations.")
    except Exception as e:
        st.error(
            f"Error fetching default recommendation for {attribute_type} - {chart_type}: {e}"
        )
        print(f"Recommendations structure: {json.dumps(recommendations, indent=4)}")
    return []


def data_visualization(DF, recommendations):
    attributes = DF.columns.tolist()
    non_unique_id_attributes = [col for col in attributes if not is_unique_id(DF, col)]

    def safe_default_index(options, default):
        if default in options:
            return options.index(default)
        return 0

    # Title for the dashboard
    st.markdown(
        """<div class="title">AI ANALYTICS DASHBOARD</div>""", unsafe_allow_html=True
    )

    # pygwalker data visualization
    with st.container():
        pyg_app = StreamlitRenderer(DF)
        pyg_app.explorer(default_tab="data", height=800)

    st.markdown(
        '<div class="black-subheader"><h3>Data Visualizations</h3></div>',
        unsafe_allow_html=True,
    )

    # Detect datetime column
    datetime_column = detect_datetime_column(DF)

    # Loop through each attribute type in recommendations
    for attribute_type, charts in recommendations.items():
        if not charts:
            continue

        st.markdown(
            f'<div class="black-subheader"><h4>{attribute_type} Visualization</h4></div>',
            unsafe_allow_html=True,
        )

        for chart_type, details in charts.items():
            attributes = details.get("Attributes", [])
            justification = details.get("Reason", "No justification provided.")
            if not attributes:
                st.warning(
                    f"No attributes found for {chart_type} in {attribute_type}. Justification: {justification}"
                )
                continue

            with st.expander(f"{chart_type} (Click to see details)"):
                st.markdown(f"**AI Justification:** {justification}")

                # Generate charts based on attribute type and chart type
                try:
                    with st.container():
                        if attribute_type == "Categorical Attribute":
                            for attribute in attributes:
                                if chart_type == "Pie Chart":
                                    fig = pie_chart(DF, attribute)
                                elif chart_type == "Bar Chart":
                                    fig = bar_chart(DF, attribute)
                                else:
                                    continue
                                st.plotly_chart(fig)

                        elif attribute_type == "Numerical Attribute":
                            for attribute in attributes:
                                if chart_type == "Distribution Histogram":
                                    fig = distribution_histogram(DF, attribute)
                                elif chart_type == "Box Plot":
                                    fig = box_plot(DF, [attribute])
                                elif chart_type == "Density Plot":
                                    fig = density_plot(DF, attribute)
                                elif chart_type == "Violin Plot":
                                    fig = violin_plot(DF, [attribute])
                                else:
                                    continue
                                st.plotly_chart(fig)

                        elif attribute_type == "Multiple Attribute":
                            if chart_type == "Heatmap":
                                fig = multi_plot_heatmap(DF, attributes)
                            elif chart_type == "Scatter Plot":
                                fig = multi_plot_scatter(DF, attributes)
                            elif chart_type == "Line Plot":
                                fig = multi_plot_line(DF, attributes)
                            elif chart_type == "Scatter Matrix":
                                fig = scatter_matrix(DF, attributes)
                            elif chart_type == "Box Plot":
                                fig = box_plot(DF, attributes)
                            elif chart_type == "Violin Plot":
                                fig = violin_plot(DF, attributes)
                            elif chart_type == "Strip Plot":
                                fig = strip_plot(DF, attributes)
                            else:
                                continue
                            st.plotly_chart(fig)

                        elif attribute_type == "3D Plot":
                            if chart_type == "3D Plot":
                                fig = scatter_3d(
                                    DF, attributes[0], attributes[1], attributes[2]
                                )
                                st.plotly_chart(fig)

                        elif attribute_type == "Word Cloud":
                            if chart_type == "Word Cloud":
                                text = DF[attributes[0]].astype(str).str.cat(sep=" ")
                                display_word_cloud(text)

                        elif attribute_type == "World Heatmap":
                            if chart_type == "World Heatmap":
                                fig = world_map(DF, attributes[0], attributes[1])
                                st.plotly_chart(fig)

                        elif attribute_type == "Time Series" and datetime_column:
                            for attribute in attributes:
                                if chart_type == "Line Plot":
                                    fig = line_plot(DF, datetime_column, attribute)
                                elif chart_type == "Area Plot":
                                    fig = area_plot(DF, datetime_column, attribute)
                                elif chart_type == "Lag Plot":
                                    fig = lag_plot(DF, attribute)
                                st.plotly_chart(fig)

                        # elif attribute_type == "Pair Plot":
                        #     if chart_type == "Pair Plot":
                        #         fig = pair_plot(DF, attributes)
                        #         st.pyplot(fig)

                        # elif attribute_type == "Radar Chart":
                        #     if chart_type == "Radar Chart":
                        #         fig = radar_chart(
                        #             DF, attributes, "group_by_column"
                        #         )  # Replace "group_by_column" with the actual column name
                        #         st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Error generating {chart_type}: {e}")

    developer_info_static()


def main():
    init_session_state()
    dataVizFile = "flag-1.csv"
    if not dataVizFile:
        st.error("No data file provided.")
        return
    if not os.path.exists(dataVizFile):
        st.error(f"File does not exist: {dataVizFile}")
        return

    file_extension = os.path.splitext(dataVizFile)[-1].lower()

    try:
        if file_extension == ".csv":
            df = pd.read_csv(dataVizFile)
        elif file_extension == ".xlsx":
            df = pd.read_excel(dataVizFile)
        else:
            st.error("Unsupported file format")
            return
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # Check if visualization recommendations are already in session state
    if (
        "visualization_recommendations" not in st.session_state
        or st.session_state["visualization_recommendations"] is None
    ):
        # ai_visual_agent(dataVizFile)
        st.session_state["visualization_recommendations"] = (
            load_visualization_recommendations()
        )

    # Check if recommendations were successfully loaded
    if st.session_state["visualization_recommendations"]:
        data_visualization(df, st.session_state["visualization_recommendations"])
    else:
        st.error("Failed to load visualization recommendations.")


if __name__ == "__main__":
    main()
