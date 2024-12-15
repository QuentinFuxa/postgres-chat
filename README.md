# RAGHandler for PostgreSQL with pgvector

A **Retrieval-Augmented Generation (RAG)** handler class built around a PostgreSQL database and OpenAI’s API. This code allows you to query a database, generate vector embeddings for columns, and integrate these operations with a conversational Large Language Model (LLM), such as GPT.

## Features

- **System Prompt Generation:** Automates the creation of a system prompt that describes your PostgreSQL table schema and sample rows.

- **Sophisticated SQL Execution**:  Runs multi-step queries with pgvector embeddings for semantic search and filtering.

- **Plotly Graph Generation**: Helps transform free-form text “drafts” into structured objects (e.g., JSON) for database insertion.


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

```
pip install -r requirements.txt
```


**Prerequisites**

-	PostgreSQL Database: You need an existing PostgreSQL database. The code connects using a provided connection string (e.g., postgresql://user:password@hostname:5432/dbname). You have to install pgvector too.

-	OpenAI API Key: Required for generating embeddings and LLM responses. You can set it via an environment variable: OPENAI_API_KEY.

## Usage

### Initialization

Use the PostgresChat class to set up the connection to your database and OpenAI.

```
from postgres_chat import PostgresChat

handler = PostgresChat(
    table_name="your_table",
    connection_string="postgresql://user:password@localhost:5432/your_database",
    openai_api_key="YOUR_OPENAI_API_KEY",
    schema="public",
    llm_model="gpt-4o",  # or any other model identifier
    embedding_model="text-embedding-3-small"  # example embedding model
)
```

## Using custom system prompt, saving and loading

When the PostgresChat is first initialized, it attempts to generate a system prompt based on your table’s columns and a sample of rows. You can also provide a custom system string using system_prompt :

```
handler = PostgresChat(
    ...
	system_prompt = ""your system prompt"""

)
```

You can also save and reload your system prompt :

```
handler.save_system_prompt(path='your_path')
```

And you can reload it using 

```
handler = PostgresChat(
    ...
	system_prompt_path = "your system prompt path"

)
```


## Adding Messages and Running the Conversation

You can start a chat with the LLM by adding user messages and then calling run_conversation():

```
handler.reinitialize_messages()  # Clears old messages. Not necessary at start

# Let's assume we have a movie database from IMBD

handler.add_user_message("""
What movies are similar to The Matrix
but have an average rating above 8.0?
Give the movie titles, ratings, and links if available.
""")
response_dict = handler.run_conversation()

print("LLM Response:", response_dict["response"])
print("Executed SQL Queries:", response_dict["executed_queries"])


handler.add_user_message("""
Generate a  chart  showing the yearly count of new releases 
from 2010 to the latest year available.
""")
response_dict = handler.run_conversation()


handler.add_user_message("""
Please list the top 5 directors with the highest average movie rating, 
alongside the average rating and the number of movies they've directed. 
Also, for each director, provide one example of their best movie link.
""")
response_dict = handler.run_conversation()

print("LLM Response:", response_dict["response"])
print("Executed SQL Queries:", response_dict["executed_queries"])

handler.add_user_message("""
Among the movies bout time travel, 
which genres have the highest average rating overall? 
Group the response by genre, include the rating, and also 
provide an example movie link from each group.
""")
response_dict = handler.run_conversation()

print("LLM Response:", response_dict["response"])
print("Executed SQL Queries:", response_dict["executed_queries"])

```

- ```response_dict["response"]```: The final textual response from the LLM.
- ```response_dict["executed_queries"]```: List of SQL queries the LLM executed under the hood.

**Executing SQL Queries Directly**

If you want to run SQL queries yourself through the PostgresChat (and automatically handle vector placeholders), you can do so directly:

```
sql_query = """
SELECT id, name, some_vector_column
FROM public.your_table
WHERE some_vector_column <-> <vector>search text<vector/> < 0.8
"""
result_string = handler.execute_sql_query(sql_query)
print("SQL Query Result:", result_string)
```

- The substring ```<vector>search text<vector/>``` will be replaced by the actual embedding array.

**Structuring Objects**

If you have a free-form “draft” text that describes an object you’d like to insert into the database, you can use:

structured_response = handler.structure_object("Draft text describing a new row or record")
print(structured_response)

This will prompt the LLM to return a structured object (like JSON) that aligns with the table’s columns.

**Creating/Embedding a Table from a DataFrame**

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

- Generates embeddings for the specified columns.
- Creates (or replaces) a table in the database with an extra column named text_column_embedding (type VECTOR(1536)).

## Environment Variables
	•	OPENAI_API_KEY: Your OpenAI API key must be set either in the environment or passed in code.

```
export OPENAI_API_KEY="sk-..."
```