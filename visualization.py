import streamlit as st
import pandas as pd
import os
from local_components import card_container
from utils import developer_info_static
from plots1 import (
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
    # radar_chart,
    scatter_matrix,
)
import pandas.api.types as ptypes
from pygwalker.api.streamlit import StreamlitRenderer


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


def data_visualization(DF):
    attributes = DF.columns.tolist()
    non_unique_id_attributes = [col for col in attributes if not is_unique_id(DF, col)]
    # Title for the dashboard
    st.markdown(
        """<div class="title">AI ANALYTICS DASHBOARD</div>""", unsafe_allow_html=True
    )

    # pygwalker data visualization
    with st.container():
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
        default_categorical_attribute = next(
            (col for col in non_unique_id_attributes if is_categorical(col, DF)),
            non_unique_id_attributes[0],
        )
        categorical_attributes = [col for col in attributes if is_categorical(col, DF)]
        categorical_attribute = st.selectbox(
            label="Select a categorical attribute to visualize:",
            options=categorical_attributes,
            index=(
                categorical_attributes.index(default_categorical_attribute)
                if default_categorical_attribute in categorical_attributes
                else 0
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
        with st.container():
            plot_area = st.empty()
            try:
                with st.container():
                    if categorical_plot_type == "Donut Chart":
                        fig = count_Y(DF, categorical_attribute)
                        plot_area.plotly_chart(fig, use_container_width=True)
                    elif categorical_plot_type == "Bar Chart":
                        fig = bar_chart(DF, categorical_attribute)
                        plot_area.plotly_chart(fig, use_container_width=True)
                    elif categorical_plot_type == "Pie Chart":
                        fig = pie_chart(DF, categorical_attribute)
                        plot_area.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating plot: {e}")
    with col_numerical:

        st.markdown(
            '<div class="black-subheader"><h4>Numerical Attribute Visualization</h4></div>',
            unsafe_allow_html=True,
        )
        default_numerical_attribute = next(
            (col for col in non_unique_id_attributes if is_numeric(col, DF)),
            non_unique_id_attributes[0],
        )
        numerical_attributes = [col for col in attributes if is_numeric(col, DF)]
        numerical_attribute = st.selectbox(
            label="Select a numerical attribute to visualize:",
            options=numerical_attributes,
            index=(
                numerical_attributes.index(default_numerical_attribute)
                if default_numerical_attribute in numerical_attributes
                else 0
            ),
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
        with st.container():
            plot_area = st.empty()
            try:
                with st.container():
                    if numerical_plot_type == "Distribution histogram":
                        fig = distribution_histogram(DF, numerical_attribute)
                        plot_area.plotly_chart(fig, use_container_width=True)
                    elif numerical_plot_type == "Boxplot":
                        fig = box_plot(DF, [numerical_attribute])
                        plot_area.plotly_chart(fig, use_container_width=True)
                    elif numerical_plot_type == "Density plot":
                        fig = density_plot(DF, numerical_attribute)
                        plot_area.plotly_chart(fig, use_container_width=True)
                    elif numerical_plot_type == "Violin plot":
                        fig = violin_plot(DF, [numerical_attribute])
                        plot_area.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating plot: {e}")

    st.divider()

    # Multiple attribute visualization
    st.markdown(
        '<div class="black-subheader"><h3>Multiple Attribute Visualization</h3></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([6, 4])
    with col1:
        default_attributes = [
            col for col in non_unique_id_attributes if is_numeric(col, DF)
        ][:3]
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
        with st.container():
            plot_area = st.empty()
    if options:
        try:
            with st.container():
                if plot_type == "Scatter plot":
                    fig = multi_plot_scatter(DF, options)
                    if fig == -1:
                        plot_area.error("Scatter plot requires two attributes")
                    else:
                        plot_area.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Heatmap":
                    fig = multi_plot_heatmap(DF, options)
                    if fig == -1:
                        plot_area.error("The attributes are not numeric")
                    else:
                        plot_area.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Boxplot":
                    fig = box_plot(DF, options)
                    if fig == -1:
                        plot_area.error("The attributes are not numeric")
                    else:
                        plot_area.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Violin plot":
                    fig = violin_plot(DF, options)
                    if fig == -1:
                        plot_area.error("The attributes are not numeric")
                    else:
                        plot_area.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Strip plot":
                    fig = strip_plot(DF, options)
                    if fig == -1:
                        plot_area.error("The attributes are not numeric")
                    else:
                        plot_area.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Line plot":
                    fig = multi_plot_line(DF, options)
                    if fig == -1:
                        plot_area.error("The attributes are not numeric")
                    elif fig == -2:
                        plot_area.error("Line plot requires two attributes")
                    else:
                        plot_area.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Radar chart":
                    fig = radar_chart(DF, options)
                    plot_area.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Scatter Matrix":
                    fig = scatter_matrix(DF, options)
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating plot: {e}")

    st.divider()

    # Advanced visualization
    st.markdown(
        '<div class="black-subheader"><h3>3D Scatter Plot</h3></div>',
        unsafe_allow_html=True,
    )
    column_1, column_2, column_3 = st.columns(3)
    default_x = (
        default_attributes[0] if default_attributes else non_unique_id_attributes[0]
    )
    default_y = (
        default_attributes[1]
        if len(default_attributes) > 1
        else non_unique_id_attributes[1]
    )
    default_z = next(
        (
            col
            for col in non_unique_id_attributes
            if is_numeric(col, DF) and col not in default_attributes
        ),
        non_unique_id_attributes[2] if len(non_unique_id_attributes) > 2 else None,
    )
    with column_1:
        x = st.selectbox(
            key="x",
            label="Select the x attribute:",
            options=attributes,
            index=attributes.index(default_x),
        )
    with column_2:
        y = st.selectbox(
            key="y",
            label="Select the y attribute:",
            options=attributes,
            index=attributes.index(default_y),
        )
    with column_3:
        z = st.selectbox(
            key="z",
            label="Select the z attribute:",
            options=attributes,
            index=attributes.index(default_z) if default_z else 0,
        )
    plot_3d_area = st.empty()  # Create an empty area for the 3D plot
    # Default 3D scatter plot
    if default_z:
        st.write(
            f"Default 3D Scatter Plot for :green[{default_x}, {default_y}, {default_z}]"
        )
        fig_3d_1 = scatter_3d(DF, default_x, default_y, default_z)
        plot_3d_area.plotly_chart(fig_3d_1, use_container_width=True)
    if st.button("Generate 3D Plot"):
        fig_3d_1 = scatter_3d(DF, x, y, z)
        if fig_3d_1 == -1:
            st.error("Data not supported")
        else:
            plot_3d_area.plotly_chart(fig_3d_1, use_container_width=True)

    st.divider()

    st.markdown(
        '<div class="black-subheader"><h3>Word Cloud</h3></div>',
        unsafe_allow_html=True,
    )

    text_attr = st.selectbox(
        label="Select the text attribute:", options=attributes, index=0
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
                st.plotly_chart(world_fig, use_container_width=True)

    with st.container():
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
        st.plotly_chart(st.session_state.overall_plot, use_container_width=True)

    developer_info_static()


def main():
    dataVizFile = "incidents.csv"
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

    data_visualization(df)


if __name__ == "__main__":
    main()
