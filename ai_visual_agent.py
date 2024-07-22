import requests
import warnings
import io
import pandas as pd
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from langchain_openai import AzureChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import json

import pandas as pd


def ai_visual_agent(file_path):
    # Import necessary libraries

    warnings.filterwarnings("ignore")

    # Enabling Azure Openai
    azure_llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        azure_endpoint="https://dwspoc.openai.azure.com/",
        api_key="6e27218cd3eb4dd0b0d94277679b79e4",
        api_version="2024-02-15-preview",
    )

    def get_csv_data(file_path, query, connection_timeout=1500):
        try:
            df = pd.read_csv(file_path)
            result = df.query(query)
            return result.to_string()
        except Exception as e:
            return f"Error processing CSV file: {e}"

    def get_excel_data(file_path, sheet_name, query, connection_timeout=1500):
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            result = df.query(query)
            return result.to_string()
        except Exception as e:
            return f"Error processing Excel file: {e}"

    class CSVQuery(BaseTool):
        name: str = "CSV Query"
        description: str = "Returns the result of query execution on CSV file"

        def _run(self, file_path: str, query: str) -> str:
            return get_csv_data(file_path, query)

    class ExcelQuery(BaseTool):
        name: str = "Excel Query"
        description: str = "Returns the result of query execution on Excel file"

        def _run(self, file_path: str, sheet_name: str, query: str) -> str:
            return get_excel_data(file_path, sheet_name, query)

    class CSVStructure(BaseTool):
        name: str = "CSV Structure"
        description: str = "Returns the list of columns and their types from a CSV file"

        def _run(self, file_path: str) -> str:
            try:
                df = pd.read_csv(file_path)
                return df.dtypes.to_string()
            except Exception as e:
                return f"Error processing CSV file: {e}"

    class CSVExamples(BaseTool):
        name: str = "CSV Examples"
        description: str = "Returns the first N rows from a CSV file"

        def _run(self, file_path: str, n: int = 30) -> str:
            try:
                df = pd.read_csv(file_path)
                return df.head(n).to_string()
            except Exception as e:
                return f"Error processing CSV file: {e}"

    class ExcelStructure(BaseTool):
        name: str = "Excel Structure"
        description: str = (
            "Returns the list of columns and their types from an Excel file"
        )

        def _run(self, file_path: str, sheet_name: str) -> str:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                return df.dtypes.to_string()
            except Exception as e:
                return f"Error processing Excel file: {e}"

    class ExcelExamples(BaseTool):
        name: str = "Excel Examples"
        description: str = "Returns the first N rows from an Excel file"

        def _run(self, file_path: str, sheet_name: str, n: int = 30) -> str:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                return df.head(n).to_string()
            except Exception as e:
                return f"Error processing Excel file: {e}"

    class DataSummary(BaseTool):
        name: str = "Data Summary"
        description: str = "Returns summary statistics of the data"

        def _run(
            self, file_path: str, file_type: str = "csv", sheet_name: str = None
        ) -> str:
            try:
                if file_type == "csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                return df.describe().to_string()
            except Exception as e:
                return f"Error processing file: {e}"

    class DataCleaning(BaseTool):
        name: str = "Data Cleaning"
        description: str = (
            "Cleans the data by handling missing values and removing duplicates"
        )

        def _run(
            self,
            file_path: str,
            file_type: str = "csv",
            sheet_name: str = None,
            fill_value: str = "mean",
        ) -> str:
            try:
                if file_type == "csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                if fill_value == "mean":
                    df.fillna(df.mean(), inplace=True)
                elif fill_value == "median":
                    df.fillna(df.median(), inplace=True)
                df.drop_duplicates(inplace=True)
                cleaned_file_path = file_path.replace(".csv", "_cleaned.csv").replace(
                    ".xlsx", "_cleaned.xlsx"
                )
                if file_type == "csv":
                    df.to_csv(cleaned_file_path, index=False)
                else:
                    df.to_excel(cleaned_file_path, index=False)
                return f"Cleaned data saved to {cleaned_file_path}"
            except Exception as e:
                return f"Error processing file: {e}"

    class DataAggregation(BaseTool):
        name: str = "Data Aggregation"
        description: str = (
            "Performs aggregation operations like sum, mean, median, etc."
        )

        def _run(
            self,
            file_path: str,
            file_type: str = "csv",
            sheet_name: str = None,
            group_by: str = None,
            agg_func: str = "mean",
        ) -> str:
            try:
                if file_type == "csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                if group_by:
                    result = df.groupby(group_by).agg(agg_func)
                else:
                    result = df.agg(agg_func)
                return result.to_string()
            except Exception as e:
                return f"Error processing file: {e}"

    class DataVisualization(BaseTool):
        name: str = "Data Visualization"
        description: str = (
            "Generates basic visualizations like histograms, scatter plots, and box plots"
        )

        def _run(
            self,
            file_path: str,
            file_type: str = "csv",
            sheet_name: str = None,
            plot_type: str = "histogram",
            column: str = None,
        ) -> str:
            try:
                if file_type == "csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                if plot_type == "histogram":
                    sns.histplot(df[column])
                elif plot_type == "scatter":
                    sns.scatterplot(data=df, x=column, y=df.columns[1])
                elif plot_type == "boxplot":
                    sns.boxplot(data=df, y=column)
                plot_file_path = file_path.replace(".csv", f"_{plot_type}.png").replace(
                    ".xlsx", f"_{plot_type}.png"
                )
                plt.savefig(plot_file_path)
                plt.close()
                return f"Plot saved to {plot_file_path}"
            except Exception as e:
                return f"Error processing file: {e}"

    class CorrelationMatrix(BaseTool):
        name: str = "Correlation Matrix"
        description: str = "Provides the correlation matrix of the data"

        def _run(
            self, file_path: str, file_type: str = "csv", sheet_name: str = None
        ) -> str:
            try:
                if file_type == "csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                corr_matrix = df.corr()
                return corr_matrix.to_string()
            except Exception as e:
                return f"Error processing file: {e}"

    # Instantiate the tools
    csv_query_tool = CSVQuery()
    excel_query_tool = ExcelQuery()
    csv_structure_tool = CSVStructure()
    csv_examples_tool = CSVExamples()
    excel_structure_tool = ExcelStructure()
    excel_examples_tool = ExcelExamples()
    data_summary_tool = DataSummary()
    data_aggregation_tool = DataAggregation()
    data_visualization_tool = DataVisualization()
    correlation_matrix_tool = CorrelationMatrix()

    # Database Specialist
    database_specialist = Agent(
        role="database specialist",
        goal="Provide a documentation of the data base and each column to understand the data better",
        backstory="""You are an expert in data analysis, so you can help the team to provide needed documentation about the data to make decision to choose fields.  
        You are very accurate and provide the precise about the data. based on your documention only data analyst choose with which visualization charts used for particular attributes and make sure the file is exist""",
        llm=azure_llm,
        verbose=True,
        allow_delegation=False,
    )

    # Visualization Expert
    visualization_expert = Agent(
        role="visualization expert",
        goal="Determine which chart is suitable for visualizing specific data attributes",
        backstory="""You are an expert in data visualization and understand the strengths and weaknesses of various chart types.
    Here are the options for different attribute types you should provide under each chart ensure the attribute or column which each chart it will accept:

1. Categorical Attribute: (chart 1)
-----------------------
Pie Chart (Accepts categorical data, 1 attribute)
Donut Chart (Accepts categorical data, 1 attribute)
Bar Chart (Accepts categorical data, 1 attribute)

2. Numerical Attribute: (chart 2)
-----------------------
Distribution Histogram (Accepts numerical data, 1 attribute)
Box Plot (Accepts numerical data, 1 attribute)
Density Plot (Accepts numerical data, 1 attribute)
Violin Plot (Accepts numerical data, 1 attribute)
2. Multiple Attribute Chart

3. Multiple Attributes: (chart 3)
--------------------------
Heatmap (Accepts numerical data, multiple attributes)
Strip Plot (Accepts numerical data, multiple attributes)
Box Plot (Accepts numerical data, multiple attributes)
2 Attribute Plot:

Line Plot (Accepts numerical data, 2 attributes; Column2 vs Column3)
Scatter Plot (Accepts numerical data, 2 attributes)
Multi Attribute Plot:

Scatter Matrix (Accepts numerical data, multiple attributes)


3. 3D Plot (chart 4)
---------------------
3 Attributes (X, Y, Z):
3D Plot (Accepts categorical or numerical data, 3 attributes)


4. Word Cloud (chart 5)
--------------------------
Word Cloud (Accepts categorical or text data, 1 attribute)
5. World Heatmap

5. Country Attribute: (chart 5)
-------------------------------
World Heatmap (Accepts country data and one additional numerical attribute; Select the attribute to display)
Using the following example data with the columns listed below, select the appropriate chart type for each category and provide a brief reason for your choice:

Example Data Columns:
Car_ID (int64): Unique identifier for each car.
Brand (object): The make of the car, e.g., Toyota, Honda.
Model (object): The specific model of the car.
Year (int64): The manufacturing year of the car.
Kilometers_Driven (int64): The total distance the car has been driven in kilometers.
Fuel_Type (object): The type of fuel the car uses, e.g., Petrol, Diesel.
Transmission (object): The type of transmission, e.g., Manual, Automatic.
Owner_Type (object): The ownership status, e.g., First, Second owner.
Mileage (int64): The fuel efficiency of the car in km/l.
Engine (int64): The engine capacity in cc.
Power (int64): The power output of the car in bhp.
Seats (int64): The number of seats in the car.
Price (int64): The price of the car in the local currency.

"Categorical Attribute": {{
    "Pie Chart": ["Brand"],
    "Reason": "Pie charts are effective for showing the proportions of different car brands in the dataset. (Accepts categorical data, 1 attribute)"
  }},
  "Numerical Attribute": {{
    "Box Plot": ["Mileage"],
    "Reason": "Box plots are ideal for displaying the distribution and identifying outliers in car mileage data. (Accepts numerical data, 1 attribute)"
  }},
  "Multiple Attribute": {{
    "Heatmap": ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"],
    "Reason": "Heatmaps provide a clear visual representation of the relationships and correlations between multiple numerical attributes. (Accepts numerical data, multiple attributes)"
  }},
  "3D Plot": {{
    "3D Plot": ["Kilometers_Driven", "Mileage", "Engine"],
    "Reason": "3D plots are suitable for visualizing the interactions between three numerical attributes simultaneously. (Accepts categorical or numerical data, 3 attributes)"
  }},
  "Word Cloud": {{
    "Word Cloud": ["Fuel_Type"],
    "Reason": "Word clouds effectively highlight the frequency of different fuel types used by cars in the dataset. (Accepts categorical or text data, 1 attribute)"
  }},
  "World Heatmap": {{
    "World Heatmap": ["Country", "Price"],
    "Reason": "World heatmaps are useful for displaying the geographical distribution of car prices across different countries. (Accepts country data and one additional numerical attribute)"
  }}

  if non of the column or attribute applicable, please ignore it in the above example. it's try to hallucinate where country column is not avaible, don't do that. if not available keep it empty
  "World Heatmap": {{
    "World Heatmap": ["Country", "Price"],
    "Reason": "World heatmaps are useful for displaying the geographical distribution of car prices across different countries. (Accepts country data and one additional numerical attribute)"
  }}
  
  "World Heatmap": {{
    "World Heatmap": ["Country", "Price"],
    "Reason": "World heatmaps are useful for displaying the geographical distribution of car prices across different countries. (Accepts country data and one additional numerical attribute)"
  }}
    Your role is to recommend the most effective chart type for visualizing specific data attributes, ensuring that the visual representation of data is clear and insightful.    
        """,
        llm=azure_llm,
        allow_delegation=False,
        verbose=True,
    )

    # Chart Attribute Formatter
    chart_attribute_formatter = Agent(
        role="chart-attribute formatter",
        goal="Provide a JSON format mapping chart types to data attributes, and generate the final output of charts",
        backstory="""You are skilled in organizing information and ensuring it is presented in a structured format.  
        Your job is to create a JSON object that maps different chart types to their respective data attributes, and produce the final visualizations based on these mappings.""",
        llm=azure_llm,
        allow_delegation=False,
        verbose=True,
    )

    table_description_task = Task(
        description="""Provide a comprehensive overview of the data in table {table} to facilitate understanding of the table's structure and contents.  
        This task is crucial for selecting the appropriate attributes for various visualization charts. The details should be passed to the visual_expert agent.""",
        expected_output="""The comprehensive overview of {table} should include the following sections:  
        1. **Columns**: A detailed list of columns with their names, data types, and a brief description of what each column represents.  
        2. **Examples**: The first 10 rows from the table to give a sample of the data contained within each column.  
        3. **Summary Statistics**: Basic summary statistics for each column, such as mean, median, mode, standard deviation, minimum, and maximum values for numeric columns; and count and unique values for categorical columns.  
        4. **Missing Values**: A report on the number and percentage of missing values in each column.""",
        tools=[
            csv_query_tool,
            excel_query_tool,
            csv_structure_tool,
            csv_examples_tool,
            excel_structure_tool,
            excel_examples_tool,
            data_summary_tool,
            data_aggregation_tool,
            correlation_matrix_tool,
        ],
        agent=database_specialist,
    )

    # Task for Visualization agent
    visualization_recommendation_task = Task(
        description="""You are an expert in data visualization and understand the strengths and weaknesses of various chart types.  
        Your task is to recommend the most effective chart type for visualizing specific data attributes in to ensure clear and insightful visual representation. Based on the provided information about the {table}, recommend the most suitable chart types for visualizing the attributes of the table.  
        The goal is to ensure that the data is presented in a clear and insightful manner.""",
        expected_output="""A JSON object that maps different chart types to their respective data attributes, including:  
        1. **Chart Type**: The recommended chart type (e.g., bar chart, scatter plot, pie chart).  
        2. **Attributes**: The data attributes that should be visualized with this chart type.  
        3. **Justification**: A brief justification for why this chart type is suitable for the selected attributes.  
        """,
        tools=[],
        output_file="visualization_recommendations.txt",
        agent=visualization_expert,
    )

    # quality analyst to check which attribute for any one of the charts to better understanding
    quality_analyst = Agent(
        role="quality analyst",
        goal="Check the appropriateness of attributes for each chart type to ensure better understanding",
        backstory="""You are a detail-oriented expert in quality analyst. Your responsibility is to review and validate the suitability of data attributes for various chart types and if any changes required delegate work to respective coworker, ensuring that the chosen visualizations recommendation provide clear and accurate insights about the {table}""",
        llm=azure_llm,
        allow_delegation=True,
        verbose=True,
    )

    # Quality Analyst Task
    quality_analysis_task = Task(
        description="""Review and validate the suitability of attributes for various chart types to ensure clear and accurate insights.  
        This task is crucial for guaranteeing that the chosen visualizations provide a better understanding of the data.""",
        expected_output="""A JSON object that maps different chart types to their respective data attributes, including:  
        1. **Chart Type**: The recommended chart type (e.g., bar chart, scatter plot, pie chart).  
        2. **Attributes**: The data attributes that should be visualized with this chart type.  
        3. **Justification**: A brief justification for why this chart type is suitable for the selected attributes.""",
        context=[table_description_task, visualization_recommendation_task],
        agent=quality_analyst,
    )

    

    # Full Crew
    full_crew = Crew(
        agents=[
            database_specialist,
            visualization_expert,
            chart_attribute_formatter,
            quality_analyst,
        ],
        tasks=[
            table_description_task,
            visualization_recommendation_task,
            quality_analysis_task,
        ],
        verbose=3,
    )

    # Kickoff the crew with the CSV file path
    full_result = full_crew.kickoff({"table": file_path})

    return full_result


# csv_file_path = r"F:\Documents backup\AI Projects\CDA_V2\CDA_V2\CDA-master 2\backend\visualization\incidents.csv"

# # Check if file exists
# if os.path.exists(csv_file_path):
#     print(os.path.exists(csv_file_path))
#     result = ai_visual_agent(csv_file_path)
# else:
#     print(f"File does not exist: {csv_file_path}")


# Read the content of the txt file
# with open("visualization_recommendations.txt", "r") as file:
#     file_content = file.read()

# # Extract JSON content using regex
# json_match = re.search(r"```json(.*?)```", file_content, re.DOTALL)
# if json_match:
#     json_content = json_match.group(1).strip()
#     print("JSON content extracted successfully.")
# else:
#     print("No JSON content found.")
#     json_content = None

# # Parse the extracted JSON content
# if json_content:
#     try:
#         visualization_recommendations = json.loads(json_content)
#         print("JSON content parsed successfully.")
#     except json.JSONDecodeError as e:
#         print(f"Error parsing JSON content: {e}")

# # Output the parsed JSON content (for demonstration purposes)
# if "visualization_recommendations" in locals():
#     print(json.dumps(visualization_recommendations, indent=4))

#     # Iterate through the JSON data and print the required information
#     for attribute_type, charts in visualization_recommendations.items():
#         print(f"{attribute_type}:")
#         for chart, details in charts.items():
#             if chart == "Reason":
#                 continue
#             attributes = details
#             reason = charts.get("Reason", "")
#             print(f"  {chart}:")
#             print(f"    Attributes: {attributes}")
#             print(f"    Reason: {reason}")
# else:
#     print("No JSON data to process.")


# # Check if file exists
# if os.path.exists(csv_file_path):
#     result = ai_visual_agent(csv_file_path)
#     # Assuming the result is a dictionary and you want to save it to a JSON file
#     with open("visualization_recommendations.json", "w") as json_file:
#         json.dump(result, json_file, indent=4)
#     print("JSON output saved to 'visualization_recommendations.json'")
# else:
#     print(f"File does not exist: {csv_file_path}")

# Load the JSON file and iterate through the data
# Load the JSON file and iterate through the data


# json_file_path = r"F:\Documents backup\AI Projects\CDA_V2\CDA_V2\CDA-master 2\backend\visualization_recommendations.json"
# if os.path.exists(json_file_path):
#     with open(json_file_path, "r") as json_file:
#         try:
#             visualization_recommendations = json.load(json_file)
#             print("JSON file loaded successfully.")
#         except json.JSONDecodeError as e:
#             print(f"Error reading JSON file: {e}")
# else:
#     print(f"JSON file does not exist: {json_file_path}")

# # Iterate through the JSON data and print the required information
# if "visualization_recommendations" in locals():
#     for attribute_type, charts in visualization_recommendations.items():
#         print(f"{attribute_type}:")
#         for chart, details in charts.items():
#             attributes = details.get("Attributes", [])
#             reason = details.get("Reason", "")
#             print(f"  {chart}:")
#             print(f"    Attributes: {attributes}")
#             print(f"    Reason: {reason}")


# import json
# import re

# # The input text containing JSON content
# input_text = """my best complete final answer to the task.
# ```json
# {
#   "Categorical Attribute": {
#     "Pie Chart": ["Brand"],
#     "Reason": "Pie charts are effective for showing the proportions of different car brands in the dataset. (Accepts categorical data, 1 attribute)"
#   },
#   "Numerical Attribute": {
#     "Box Plot": ["Mileage"],
#     "Reason": "Box plots are ideal for displaying the distribution and identifying outliers in car mileage data. (Accepts numerical data, 1 attribute)"
#   },
#   "Multiple Attribute": {
#     "Heatmap": ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"],
#     "Reason": "Heatmaps provide a clear visual representation of the relationships and correlations between multiple numerical attributes. (Accepts numerical data, multiple attributes)"
#   },
#   "3D Plot": {
#     "3D Plot": ["Kilometers_Driven", "Mileage", "Engine"],
#     "Reason": "3D plots are suitable for visualizing the interactions between three numerical attributes simultaneously. (Accepts categorical or numerical data, 3 attributes)"
#   },
#   "Word Cloud": {
#     "Word Cloud": ["Fuel_Type"],
#     "Reason": "Word clouds effectively highlight the frequency of different fuel types used by cars in the dataset. (Accepts categorical or text data, 1 attribute)"
#   },
#   "World Heatmap": {
#     "World Heatmap": [],
#     "Reason": "World heatmaps are useful for displaying the geographical distribution of car prices across different countries. (Accepts country data and one additional numerical attribute). However, the dataset does not contain country data."
#   }
# }
# ```This JSON object provides a comprehensive recommendation for the most suitable chart types for visualizing specific data attributes in the provided "cars" table, ensuring clear and insightful visual representation."""

# # Extract JSON content using regex
# json_match = re.search(r"```json(.*?)```", input_text, re.DOTALL)
# if json_match:
#     json_content = json_match.group(1).strip()
# else:
#     print("No JSON content found.")

# # Parse the extracted JSON content
# try:
#     visualization_recommendations = json.loads(json_content)
#     print("JSON content extracted and parsed successfully.")
# except json.JSONDecodeError as e:
#     print(f"Error parsing JSON content: {e}")

# # Output the parsed JSON content (for demonstration purposes)
# print(json.dumps(visualization_recommendations, indent=4))

# # Iterate through the JSON data and print the required information
# for attribute_type, charts in visualization_recommendations.items():
#     print(f"{attribute_type}:")
#     for chart, details in charts.items():
#         if chart == "Reason":
#             continue
#         attributes = details
#         reason = charts.get("Reason", "")
#         print(f"  {chart}:")
#         print(f"    Attributes: {attributes}")
#         print(f"    Reason: {reason}")
