# RAGHandler for PostgreSQL

A **Retrieval-Augmented Generation (RAG)** handler class built around a PostgreSQL database and OpenAI’s API. This code allows you to query a database, generate vector embeddings for columns, and integrate these operations with a conversational Large Language Model (LLM), such as GPT.

## Features

	-	**System Prompt Generation:** Automates the creation of a system prompt that describes your PostgreSQL table schema and sample rows.
	-	**SQL Execution & Vector Replacement: Executes SQL queries directly against the database. If the SQL query contains <vector>TEXT<vector/> placeholders, those placeholders are replaced with vector embeddings generated via OpenAI.
	-	**Chat-Based Interaction:** Supports a chat-based workflow by combining user messages, system prompts, and LLM responses. Integrates function calls for:
	    -	Executing SQL queries (execute_sql_query),
	    -	Structuring objects for potential insertion (structure_object_from_draft),
	    -	Generating Plotly graphs (plot_graph).
	-	**Schema Summarization**: Summarizes a database table’s schema by prompting the LLM, providing a quick overview of column purposes and example values.
	-	**Object Structuring**: Helps transform free-form text “drafts” into structured objects (e.g., JSON) for database insertion.
	-	**Embedding Integration**: Automatically generates and stores vector embeddings for specified text columns when creating or replacing tables.


## Installation

```
# Clone the repository
git clone https://github.com/QuentinFuxa/postgres-chat

# Navigate into the repo
cd postgres-chat

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/MacOS
# or:
# .\venv\Scripts\activate  # Windows
```


## Install dependencies
pip install -r requirements.txt

Dependencies :
	•	openai
	•	pandas
	•	psycopg2
	•	SQLAlchemy
	•	plotly (if you need graphing functionality)

Prerequisites
	1.	PostgreSQL Database: You need an existing PostgreSQL database. The code connects using a provided connection string (e.g., postgresql://user:password@hostname:5432/dbname).
	2.	OpenAI API Key: Required for generating embeddings and LLM responses. You can set it via an environment variable: OPENAI_API_KEY.

## Usage

### Initialization

Use the RAGHandler class to set up the connection to your database and OpenAI.

```
from rag_handler import RAGHandler

handler = RAGHandler(
    table_name="your_table",
    connection_string="postgresql://user:password@localhost:5432/your_database",
    openai_api_key="YOUR_OPENAI_API_KEY",
    schema="public",
    llm_model="gpt-4o",  # or any other model identifier
    embedding_model="text-embedding-3-small"  # example embedding model
)
```

	•	table_name: The target PostgreSQL table name you want to interact with.
	•	connection_string: The PostgreSQL connection string.
	•	openai_api_key: Your OpenAI API key (can also be set as an environment variable).
	•	schema: Optional, defaults to "public".
	•	llm_model: Which LLM model to use for chat completions.
	•	embedding_model: Which model to use for embeddings.


## Generating the System Prompt

When the RAGHandler is first initialized, it attempts to generate a system prompt based on your table’s columns and a sample of rows. This system prompt is stored in handler.system_prompt.

If needed, you can regenerate or overwrite the system prompt:

```
handler._generate_system_prompt()
print(handler.system_prompt)
```

## Adding Messages and Running the Conversation

You can simulate a chat with the LLM by adding user messages and then calling run_conversation():

```
handler.reinitialize_messages()  # Clears old messages, loads system prompt

handler.add_user_message("Hello, can you give me a summary of the data?")
response_dict = handler.run_conversation()

print("LLM Response:", response_dict["response"])
print("Executed SQL Queries:", response_dict["executed_queries"])
```

	•	response_dict["response"]: The final textual response from the LLM.
	•	response_dict["executed_queries"]: List of SQL queries the LLM executed under the hood.

Executing SQL Queries Directly

If you want to run SQL queries yourself through the RAGHandler (and automatically handle vector placeholders), you can do so directly:

sql_query = """
SELECT id, name, some_vector_column
FROM public.your_table
WHERE some_vector_column <-> <vector>search text<vector/> < 0.8
"""
result_string = handler.execute_sql_query(sql_query)
print("SQL Query Result:", result_string)

	•	The substring <vector>search text<vector/> will be replaced by the actual embedding array.

Structuring Objects

If you have a free-form “draft” text that describes an object you’d like to insert into the database, you can use:

structured_response = handler.structure_object("Draft text describing a new row or record")
print(structured_response)

This will prompt the LLM to return a structured object (like JSON) that aligns with the table’s columns.

Creating/Embedding a Table from a DataFrame

You can create or replace a table from a pandas DataFrame. Specify which columns need vector embeddings:

```
import pandas as pd

# Example DataFrame
df = pd.DataFrame({
    "id": [1, 2, 3],
    "text_column": ["Hello world", "Another row", "More text"]
})

# Create or replace table, embedding 'text_column'
handler.create_table_from_df(df, embed_columns=["text_column"], table_name="new_table")
```

This:
	1.	Generates embeddings for the specified columns.
	2.	Creates (or replaces) a table in the database with an extra column named text_column_embedding (type VECTOR(1536)).

## Environment Variables
	•	OPENAI_API_KEY: Your OpenAI API key must be set either in the environment or passed in code.

```
export OPENAI_API_KEY="sk-..."
```