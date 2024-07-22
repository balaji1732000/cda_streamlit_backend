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
)
import pandas.api.types as ptypes
from pygwalker.api.streamlit import StreamlitRenderer

from ai_visual_agent import ai_visual_agent


# Function to identify unique ID columns
def is_unique_id(df, column, threshold=0.9):
    return df[column].nunique() / len(df) > threshold


def init_session_state():
    if "selected_functionality" not in st.session_state:
        st.session_state["selected_functionality"] = None


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
                    st.error(
                        f"Attributes for {chart_type} in {attribute_type} is not a list or is empty."
                    )
            else:
                st.error(f"Chart type {chart_type} not found in {attribute_type}.")
        else:
            st.error(f"Attribute type {attribute_type} not found in recommendations.")
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
    with st.container(border=True):
        pyg_app = StreamlitRenderer(DF)
        pyg_app.explorer(default_tab="data", height=800)

    st.markdown(
        '<div class="black-subheader"><h3>Single Attribute Visualization</h3></div>',
        unsafe_allow_html=True,
    )
    # Create two columns for categorical and numerical attributes and their visualizations
    col_category, col_numerical = st.columns(2)
    with col_category:
        st.markdown(
            '<div class="black-subheader"><h4>Categorical Attribute Visualization</h4></div>',
            unsafe_allow_html=True,
        )
        default_categorical_attribute = (
            get_default_recommendation(
                recommendations, "Categorical Attribute", "Pie Chart"
            )[0]
            if get_default_recommendation(
                recommendations, "Categorical Attribute", "Pie Chart"
            )
            else (
                non_unique_id_attributes[0]
                if non_unique_id_attributes
                else attributes[0]
            )
        )
        categorical_options = [col for col in attributes if is_categorical(col, DF)]
        categorical_attribute = st.selectbox(
            label="Select a categorical attribute to visualize:",
            options=categorical_options,
            index=safe_default_index(
                categorical_options, default_categorical_attribute
            ),
        )
        st.write(f"Categorical Attribute selected: :green[{categorical_attribute}]")
        categorical_plot_types = ["Donut Chart", "Bar Chart", "Pie Chart"]
        categorical_plot_type = st.selectbox(
            key="categorical_plot_type",
            label="Select a plot type for categorical attribute:",
            options=categorical_plot_types,
            index=0,
        )
        st.write(f"Categorical Plot type selected: :green[{categorical_plot_type}]")
        with st.container(border=True):
            plot_area = st.empty()
            try:
                with st.container(border=True):
                    if categorical_plot_type == "Donut Chart":
                        fig = count_Y(DF, categorical_attribute)
                        plot_area.plotly_chart(fig)
                    elif categorical_plot_type == "Bar Chart":
                        fig = bar_chart(DF, categorical_attribute)
                        plot_area.plotly_chart(fig)
                    elif categorical_plot_type == "Pie Chart":
                        fig = pie_chart(DF, categorical_attribute)
                        plot_area.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error generating plot: {e}")
    with col_numerical:
        st.markdown(
            '<div class="black-subheader"><h4>Numerical Attribute Visualization</h4></div>',
            unsafe_allow_html=True,
        )
        default_numerical_attribute = (
            get_default_recommendation(
                recommendations, "Numerical Attribute", "Box Plot"
            )[0]
            if get_default_recommendation(
                recommendations, "Numerical Attribute", "Box Plot"
            )
            else (
                non_unique_id_attributes[0]
                if non_unique_id_attributes
                else attributes[0]
            )
        )
        numerical_options = [col for col in attributes if is_numeric(col, DF)]
        numerical_attribute = st.selectbox(
            label="Select a numerical attribute to visualize:",
            options=numerical_options,
            index=safe_default_index(numerical_options, default_numerical_attribute),
        )
        st.write(f"Numerical Attribute selected: :green[{numerical_attribute}]")
        numerical_plot_types = [
            "Distribution histogram",
            "Boxplot",
            "Density plot",
            "Violin plot",
        ]
        numerical_plot_type = st.selectbox(
            key="numerical_plot_type",
            label="Select a plot type for numerical attribute:",
            options=numerical_plot_types,
            index=0,
        )
        st.write(f"Numerical Plot type selected: :green[{numerical_plot_type}]")
        with st.container(border=True):
            plot_area = st.empty()
            try:
                with st.container(border=True):
                    if numerical_plot_type == "Distribution histogram":
                        fig = distribution_histogram(DF, numerical_attribute)
                        plot_area.plotly_chart(fig)
                    elif numerical_plot_type == "Boxplot":
                        fig = box_plot(DF, [numerical_attribute])
                        plot_area.plotly_chart(fig)
                    elif numerical_plot_type == "Density plot":
                        fig = density_plot(DF, numerical_attribute)
                        plot_area.plotly_chart(fig)
                    elif numerical_plot_type == "Violin plot":
                        fig = violin_plot(DF, [numerical_attribute])
                        plot_area.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error generating plot: {e}")

    st.divider()

    st.markdown(
        '<div class="black-subheader"><h3>Multiple Attribute Visualization</h3></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([6, 4])
    with col1:
        default_attributes = (
            get_default_recommendation(recommendations, "Multiple Attribute", "Heatmap")
            or [col for col in non_unique_id_attributes if is_numeric(col, DF)][:3]
        )
        options = st.multiselect(
            label="Select multiple attributes to visualize:",
            options=attributes,
            default=default_attributes,
        )
    with col2:
        plot_types = [
            "Heatmap",
            "Strip plot",
            "Line plot",
            "Scatter plot",
            "Radar chart",
            "Scatter Matrix",
            "Boxplot",
        ]
        plot_type = st.selectbox(
            key="plot_type2",
            label="Select a plot type:",
            options=plot_types,
            index=0,
        )

    _, col_mid, _ = st.columns([1, 5, 1])
    with col_mid:
        with st.container(border=True):
            plot_area = st.empty()
    if options:
        try:
            with st.container(border=True):
                if plot_type == "Scatter plot":
                    fig = multi_plot_scatter(DF, options)
                    if fig == -1:
                        plot_area.error("Scatter plot requires two attributes")
                    else:
                        plot_area.plotly_chart(fig)
                elif plot_type == "Heatmap":
                    fig = multi_plot_heatmap(DF, options)
                    if fig == -1:
                        plot_area.error("The attributes are not numeric")
                    else:
                        plot_area.plotly_chart(fig)
                elif plot_type == "Boxplot":
                    fig = box_plot(DF, options)
                    if fig == -1:
                        plot_area.error("The attributes are not numeric")
                    else:
                        plot_area.plotly_chart(fig)
                elif plot_type == "Violin plot":
                    fig = violin_plot(DF, options)
                    if fig == -1:
                        plot_area.error("The attributes are not numeric")
                    else:
                        plot_area.plotly_chart(fig)
                elif plot_type == "Strip plot":
                    fig = strip_plot(DF, options)
                    if fig == -1:
                        plot_area.error("The attributes are not numeric")
                    else:
                        plot_area.plotly_chart(fig)
                elif plot_type == "Line plot":
                    fig = multi_plot_line(DF, options)
                    if fig == -1:
                        plot_area.error("The attributes are not numeric")
                    elif fig == -2:
                        plot_area.error("Line plot requires two attributes")
                    else:
                        plot_area.plotly_chart(fig)
                elif plot_type == "Radar chart":
                    fig = radar_chart(DF, options)
                    plot_area.plotly_chart(fig)
                elif plot_type == "Scatter Matrix":
                    fig = scatter_matrix(DF, options)
                    st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error generating plot: {e}")

    st.divider()

    # Advanced visualization
    st.markdown(
        '<div class="black-subheader"><h3>3D Scatter Plot</h3></div>',
        unsafe_allow_html=True,
    )
    column_1, column_2, column_3 = st.columns(3)
    default_x, default_y, default_z = get_default_recommendation(
        recommendations, "3D Plot", "3D Plot"
    ) or [
        (default_attributes[0] if default_attributes else non_unique_id_attributes[0]),
        (
            default_attributes[1]
            if len(default_attributes) > 1
            else non_unique_id_attributes[1]
        ),
        next(
            (
                col
                for col in non_unique_id_attributes
                if is_numeric(col, DF) and col not in default_attributes
            ),
            non_unique_id_attributes[2] if len(non_unique_id_attributes) > 2 else None,
        ),
    ]
    with column_1:
        x = st.selectbox(
            key="x",
            label="Select the x attribute:",
            options=attributes,
            index=safe_default_index(attributes, default_x),
        )
    with column_2:
        y = st.selectbox(
            key="y",
            label="Select the y attribute:",
            options=attributes,
            index=safe_default_index(attributes, default_y),
        )
    with column_3:
        z = st.selectbox(
            key="z",
            label="Select the z attribute:",
            options=attributes,
            index=safe_default_index(attributes, default_z) if default_z else 0,
        )
    plot_3d_area = st.empty()  # Create an empty area for the 3D plot
    # Default 3D scatter plot
    if default_z:
        st.write(
            f"Default 3D Scatter Plot for :green[{default_x}, {default_y}, {default_z}]"
        )
        fig_3d_1 = scatter_3d(DF, default_x, default_y, default_z)
        plot_3d_area.plotly_chart(fig_3d_1)
    if st.button("Generate 3D Plot"):
        fig_3d_1 = scatter_3d(DF, x, y, z)
        if fig_3d_1 == -1:
            st.error("Data not supported")
        else:
            plot_3d_area.plotly_chart(fig_3d_1)

    st.divider()

    st.markdown(
        '<div class="black-subheader"><h3>Word Cloud</h3></div>',
        unsafe_allow_html=True,
    )

    text_attr = (
        get_default_recommendation(recommendations, "Word Cloud", "Word Cloud")[0]
        if get_default_recommendation(recommendations, "Word Cloud", "Word Cloud")
        else (
            non_unique_id_attributes[0] if non_unique_id_attributes else attributes[0]
        )
    )
    text_attr = st.selectbox(
        label="Select the text attribute:",
        options=attributes,
        index=safe_default_index(attributes, text_attr),
    )
    if st.button("Generate Word Cloud"):
        text = DF[text_attr].astype(str).str.cat(sep=" ")
        display_word_cloud(text)

    st.divider()

    st.markdown(
        '<div class="black-subheader"><h3>World Heat Map</h3></div>',
        unsafe_allow_html=True,
    )
    col_1, col_2 = st.columns(2)
    with col_1:
        country_col = st.selectbox(
            key="country_col",
            label="Select the country attribute:",
            options=attributes,
            index=0,
        )
    with col_2:
        heat_attribute = st.selectbox(
            key="heat_attribute",
            label="Select the attribute to display in heat map:",
            options=attributes,
            index=len(attributes) - 1,
        )
    if st.button("Show Heatmap"):
        _, map_col, _ = st.columns([1, 3, 1])
        with map_col:
            world_fig = world_map(DF, country_col, heat_attribute)
            if world_fig == -1:
                st.error("Data not supported")
            else:
                st.plotly_chart(world_fig)

    with st.container(border=True):
        # Data Overview
        st.markdown(
            '<div class="black-subheader"><h3>Data Overview</h3></div>',
            unsafe_allow_html=True,
        )
        if "data_origin" not in st.session_state:
            st.session_state.data_origin = DF
        st.dataframe(st.session_state.data_origin.describe(), width=1200)
        if "overall_plot" not in st.session_state:
            st.session_state.overall_plot = list_all(st.session_state.data_origin)
        st.plotly_chart(st.session_state.overall_plot)

    developer_info_static()


def main():
    dataVizFile = "cars.csv"
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

    # Generate visualization recommendations
    # ai_visual_agent(dataVizFile)
    recommendations = load_visualization_recommendations()

    # Check if recommendations were successfully loaded
    if recommendations:
        data_visualization(df, recommendations)
    else:
        st.error("Failed to load visualization recommendations.")


if __name__ == "__main__":
    main()
