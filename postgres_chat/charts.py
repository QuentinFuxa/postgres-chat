import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

def create_graph(
    query,
    engine,                # SQLAlchemy engine or connection string
    chart_type='bar',      # chart type (bar, scatter, line, pie, etc.)
    x_col=None,            # column name for the X-axis
    y_col=None,            # column name for the Y-axis
    color_col=None,        # column name for color encoding (categories)
    size_col=None,         # column name for size encoding (e.g., scatter bubble chart)
    title='My Graph',      # chart title
    width=800,             # chart width
    height=600,            # chart height
    orientation='v',       # orientation 'v' or 'h' for bar charts
    labels=None,           # dictionary for renaming axis labels or legend labels
    template='plotly_white' # chart style template
):
    """
    Executes the SQL query, stores the result in a Pandas DataFrame, 
    and generates a Plotly chart as JSON.
    """

    # 1) Execute the SQL query
    df = pd.read_sql(query, con=engine)

    # Verify we have a non-empty DataFrame
    if df.empty:
        raise ValueError("The SQL query returned no results or the DataFrame is empty.")

    # Verify specified columns exist
    if x_col and x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' does not exist in the DataFrame.")
    if y_col and y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' does not exist in the DataFrame.")
    if color_col and color_col not in df.columns:
        raise ValueError(f"Column '{color_col}' does not exist in the DataFrame.")
    if size_col and size_col not in df.columns:
        raise ValueError(f"Column '{size_col}' does not exist in the DataFrame.")

    fig = None

    # 2) Generate the Plotly chart based on chart_type
    if chart_type == 'bar':
        # Bar chart
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            orientation=orientation,
            labels=labels,
            title=title,
            template=template,
            width=width,
            height=height
        )
    elif chart_type == 'scatter':
        # Scatter (point cloud)
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            labels=labels,
            title=title,
            template=template,
            width=width,
            height=height
        )
    elif chart_type == 'line':
        # Line chart
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            labels=labels,
            title=title,
            template=template,
            width=width,
            height=height
        )
    elif chart_type == 'pie':
        # Pie chart
        if not x_col:
            raise ValueError("For a pie chart, please specify 'x_col' for category names (or 'values' for quantities).")
        fig = px.pie(
            df,
            names=x_col,
            values=y_col,
            color=color_col,
            title=title,
            template=template,
            width=width,
            height=height
        )
    else:
        # Custom chart type or fallback
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            orientation=orientation,
            labels=labels,
            title=f"Chart type '{chart_type}' not handled, defaulting to bar chart.",
            template=template,
            width=width,
            height=height
        )

    # 3) Extra customization via fig.update_layout or fig.update_traces if needed
    fig.update_layout(
        showlegend=True,
        legend=dict(title=''),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # 4) Export figure to JSON for frontend rendering
    fig_json = fig.to_json()
    return fig_json, fig