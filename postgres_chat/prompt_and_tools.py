BASE_SYSTEM_PROMPT = """
You have access to a PostgreSQL database. Below is a description of the columns

### Table Name
{table_name}

### Table Columns
{columns_synthesis}

You will assist users in querying the data and providing insights.

### Instructions
1. The database may contain any table structure. Use the description provided above to understand the data.
2. A user will ask you questions about this database. You have two main actions:
    - Execute SQL queries to fetch the required data.
    - Respond directly to the user based on the query results.
3. **Do not mention using or a database explicitly in your response to the user.** Instead, provide the answer in plain language. For example:
   - User: "How many xxx are there?"
   - Response: "There are 1500 xxx"
4. When querying data:
    - If the table contains a column for URLs or any other reference link, always include it in your SELECT statement.
    - Summarize or group data where appropriate to avoid overly granular results.
    - Use logical defaults for time periods (e.g., group data by year or select recent records).
5. Vectorization: 
    - When a term or phrase in the user query requires a vector comparison, use the format <vector>TEXT TO VECTORIZE<vector/> where TEXT TO VECTORIZE represents the text to be vectorized.
    - Use this on the compatible columns, instead of LIKE or ILIKE.

### Additional Notes
1. If the user provides incomplete information, use logical assumptions based on the data schema.
2. Group or aggregate data appropriately for readability in the response.
3. Always ensure your responses are user-friendly and concise.
"""

STRUCTURE_OBJECT_PROMPT = """
You have access to a PostgreSQL database. Below is a description of the columns

### Table Name
{table_name}

### Table Columns
{columns_synthesis}

Your task is to structure the object in the following text, in order to potentially insert it into the database.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": "Execute a provided SQL query against the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "The SQL query to execute.",
                    },
                },
                "required": ["sql_query"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "structure_object_from_draft",
            "description": "Structure an object from a draft text for potential database insertion",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft": {
                        "type": "string",
                        "description": "The draft text to structure.",
                    },
                },
                "required": ["draft"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_graph",
            "description": "Generate a Plotly graph from JSON data",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_graph": {
                        "type": "string",
                        "description": "The JSON data for plotting.",
                    },
                },
                "required": ["json_graph"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_graph",
            "description": "Generate a Plotly chart (as JSON) from a SQL query and chart parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to be executed for data retrieval."
                    },
                    "chart_type": {
                        "type": "string",
                        "description": "Type of the chart to generate (e.g., bar, scatter, line, pie, etc.).",
                        "enum": ["bar", "scatter", "line", "pie"]
                    },
                    "x_col": {
                        "type": "string",
                        "description": "Column name to use for the X-axis."
                    },
                    "y_col": {
                        "type": "string",
                        "description": "Column name to use for the Y-axis."
                    },
                    "color_col": {
                        "type": "string",
                        "description": "Column name for color grouping (categorical)."
                    },
                    "size_col": {
                        "type": "string",
                        "description": "Column name for marker size (e.g., bubble charts)."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the generated chart."
                    },
                    "width": {
                        "type": "number",
                        "description": "Width of the chart in pixels."
                    },
                    "height": {
                        "type": "number",
                        "description": "Height of the chart in pixels."
                    },
                    "orientation": {
                        "type": "string",
                        "description": "Orientation for bar charts: 'v' or 'h'.",
                        "enum": ["v", "h"]
                    },
                    "labels": {
                        "type": "object",
                        "description": "Dictionary for axis/legend label customization."
                    },
                    "template": {
                        "type": "string",
                        "description": "Plotly template for chart styling (e.g., 'plotly_white')."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }
]