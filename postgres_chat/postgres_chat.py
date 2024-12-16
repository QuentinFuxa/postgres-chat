import os
import json
import re
import pandas as pd
from sqlalchemy import create_engine, make_url
from sqlalchemy.types import UserDefinedType
import psycopg2
from openai import OpenAI
from typing import List, Dict, Any
from prompt_and_tools import BASE_SYSTEM_PROMPT, STRUCTURE_OBJECT_PROMPT, TOOLS
from charts import create_graph

class VECTOR(UserDefinedType):
    """
    Custom SQLAlchemy type for storing embeddings in a PostgreSQL table.
    The '1536' dimension size is model-specific.
    """
    def get_col_spec(self):
        return "VECTOR(1536)"


class PostgresChat:
    """
    Retrieval-Augmented Generation Handler that wraps all logic needed
    to talk to a PostgreSQL database via LLM queries.
    """

    def __init__(
        self,
        table_name: str,
        connection_string: str,
        openai_api_key: str = None,
        schema: str = "public",
        llm_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        system_prompt: str = None,
        system_prompt_path: str = None
        
    ):
        """
        Initializes a RAGHandler instance with database and LLM configurations.

        Args:
            table_name (str): Name of the primary table to interact with.
            connection_string (str): PostgreSQL connection string.
            openai_api_key (str, optional): OpenAI API key. 
                If None, uses the OPENAI_API_KEY environment variable. Defaults to None.
            schema (str, optional): PostgreSQL schema name. Defaults to "public".
            llm_model (str, optional): Chat-completion model identifier. Defaults to "gpt-4o".
            embedding_model (str, optional): Embedding model identifier. Defaults to "text-embedding-3-small".
            system_prompt (str, optional): Custom system prompt string. If None, a new prompt 
                is generated from the schema. Defaults to None.
            system_prompt_path (str, optional): Path to a file containing a custom system prompt. 
                Used if system_prompt is None. Defaults to None.

        Raises:
            ValueError: If no valid OpenAI API key is provided (either explicitly or via environment variable).

        Side Effects:
            - Establishes a connection to the PostgreSQL database using psycopg2 and SQLAlchemy.
            - Generates or loads a system prompt.
            - Sets up an OpenAI client for embeddings and chat completions.

        Example:
            >>> handler = RAGHandler(
            ...     table_name="movies",
            ...     connection_string="postgresql://user:pass@localhost:5432/mydb",
            ...     openai_api_key="sk-...",
            ...     schema="public",
            ...     llm_model="gpt-4o",
            ...     embedding_model="text-embedding-3-small"
            ... )
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.schema = schema
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.system_prompt = system_prompt

        # Set up engine and cursor
        self.url = make_url(connection_string)
        self.engine = create_engine(self.url)
        self.connection = psycopg2.connect(connection_string)
        self.cursor = self.connection.cursor()

        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key is None:
                raise ValueError("OpenAI API key must be provided or set as an environment variable.")

        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)

        # Prepare conversation messages (system + user)
        self.messages: List[Dict[str, Any]] = []

        # Generate the initial system prompt
        if self.system_prompt is None:
            if system_prompt_path is not None:
                self.load_system_prompt(system_prompt_path)
            else:
                self._generate_system_prompt()
        
        self.messages.append({
            "role": "system",
            "content": self.system_prompt
        })

    def load_system_prompt(self, path: str):
        """
        Loads the system prompt from a file.
        """
        with open(path, "r") as f:
            self.system_prompt = f.read()

    def _generate_table_schema_and_sample(self):
        """
        Generates a description of the table schema and retrieves sample rows.
        """
        with self.engine.connect() as conn:
            schema_query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{self.table_name}'
            AND table_schema = '{self.schema}'
            """
            schema_df = pd.read_sql(schema_query, conn)

            sample_query = f"""
            SELECT * FROM {self.schema}.{self.table_name} LIMIT 10
            """
            sample_rows = pd.read_sql(sample_query, conn)

        # Build a human-friendly schema description
        schema_description = (
            "The table contains the following columns:\n" +
            "\n".join([
                f" - {row['column_name']}: Type: {row['data_type']}"
                for _, row in schema_df.iterrows()
            ])
        )
        sample_data = sample_rows.to_dict(orient="records")
        return schema_description, sample_data


    def _summarize_table_columns(self, schema_description: str, sample_data: List[Dict[str, Any]]) -> str:
        """
        Queries the LLM to generate a synthesis or summary for each column based on
        the schema description and sample data.
        """
        prompt = f"""
            You have the following database table schema:
            {schema_description}
            
            Here are 5 sample rows (as JSON-like objects):
            {sample_data}
            
            Please provide a concise synthesis of each column. For example, describe the column's potential meaning or use.
            Provide also the type of data it contains, with some examples if possible.
            Remember that you are only provided with a limited sample of the data, so don't make crazy assumptions. If you don't understand a column, you don't need to provide a summary for it.
            Be clear and structured in your summary.
            
            Return a summary in this format:
            
            1. Column Name: 
                - Type of data.
                - Summary of the column.
                - Example values: value1, ...
            2. ...

        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # smaller model for summarization
            messages=[
                {"role": "system", "content": "You are a helpful assistant that understands data schemas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return response.choices[0].message.content


    def _generate_system_prompt(self):
        """
        Generates the initial system prompt describing the table structure and columns.
        """
        schema_description, sample_data = self._generate_table_schema_and_sample()
        columns_synthesis = self._summarize_table_columns(schema_description, sample_data)

        self.system_prompt = BASE_SYSTEM_PROMPT.format(
            table_name=f"{self.schema}.{self.table_name}",
            columns_synthesis=columns_synthesis
        )


    def _replace_vector_placeholders(self, sql_query: str) -> str:
        """
        Replaces <vector>TEXT<vector/> placeholders with embeddings in SQL queries.
        """
        placeholders = re.findall(r"<vector>(.*?)<vector/>", sql_query)
        for text in placeholders:
            embedding = self._generate_embedding(text)
            embedding_str = "ARRAY[" + ", ".join(map(str, embedding)) + "]::vector"
            sql_query = sql_query.replace(f"<vector>{text}<vector/>", embedding_str)
        return sql_query


    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for a given text using the specified OpenAI model.
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding


    def execute_sql_query(self, sql_query: str) -> str:
        """
        Executes an SQL query after converting <vector> placeholders into actual vector embeddings.
        Returns the query results or an error message.
        """
        sql_query = self._replace_vector_placeholders(sql_query)
        try:
            self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            return str(results)
        except Exception as e:
            return f"Error executing SQL query: {str(e)}"


    def llm_prompt(self, conversation_messages: List[Dict[str, Any]]) -> Any:
        """
        Generic function to send messages to the LLM using Chat Completion.
        Includes the custom 'TOOLS' spec for function calls.
        """
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=conversation_messages,
            temperature=0,
            tools=TOOLS,
        )
        return response


    def structure_object(self, draft: str) -> str:
        """
        Uses the LLM to structure an object (JSON or dictionary) for potential DB insertion.
        """
        schema_description, _ = self._generate_table_schema_and_sample()

        structured_object_prompt = STRUCTURE_OBJECT_PROMPT.format(
            table_name=f"{self.schema}.{self.table_name}",
            columns_synthesis=schema_description
        )

        messages = [
            {"role": "system", "content": structured_object_prompt},
            {"role": "user", "content": draft}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content


    def create_table_from_df(
        self,
        df: pd.DataFrame,
        embed_columns: List[str] = None,
        table_name: str = None
    ):
        """
        Creates or replaces a table from a pandas DataFrame, embedding specified columns.
        """
        if embed_columns is None:
            embed_columns = []

        target_table = table_name if table_name else self.table_name
        dtypes = {}

        # Generate embeddings for specified columns
        for column in embed_columns:
            if column not in df.columns:
                continue
            df[column + '_embedding'] = df[column].apply(
                lambda x: self._generate_embedding(x) if pd.notnull(x) else None
            )
            dtypes[column + '_embedding'] = VECTOR

        df.to_sql(
            name=target_table,
            con=self.engine,
            schema=self.schema,
            if_exists='replace',
            index=False,
            dtype=dtypes
        )


    def add_user_message(self, user_content: str):
        """
        Adds a user message to the conversation history.
        """
        self.messages.append({
            "role": "user",
            "content": user_content
        })

    def reinitialize_messages(self):
        """
        Reinitializes the conversation messages to only include the system prompt.
        """
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        
    def save_system_prompt(self, path: str):
        """
        Saves the system prompt to a file.
        """
        with open(path, "w") as f:
            f.write(self.system_prompt)
        

    def run_conversation(self) -> dict:
        """
        Executes the conversation flow until the LLM stops making tool calls 
        and provides a final response.
        """
        list_of_executed_queries = []
        response_text = ""

        while True:
            # Call the LLM with the current conversation
            response_raw = self.llm_prompt(self.messages)
            assistant_message = response_raw.choices[0].message
            self.messages.append(assistant_message)

            tool_calls = assistant_message.tool_calls

            if tool_calls and len(tool_calls) > 0:
                # Iterate over each tool call
                for idx, tool_call in enumerate(tool_calls):
                    tool_arguments = json.loads(tool_call.function.arguments)

                    if tool_call.function.name == "execute_sql_query":
                        sql_query = tool_arguments["sql_query"]
                        list_of_executed_queries.append(sql_query)

                        sql_response = self.execute_sql_query(sql_query)
                        # Summarize or truncate if needed
                        if len(sql_response) > 50000:
                            sql_response = "Query result is too long. Please try a simpler query."
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": sql_response
                        })

                    elif tool_call.function.name == "structure_object_from_draft":
                        draft_text = tool_arguments["draft"]
                        response_text = self.structure_object(draft_text)
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": response_text
                        })

                    elif tool_call.function.name == "plot_graph":
                        json_graph = tool_arguments["json_graph"]
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json_graph
                        })

                    elif tool_call.function.name == "create_graph":
                        # Parse chart creation arguments
                        query = tool_arguments["query"]
                        chart_type = tool_arguments.get("chart_type", "bar")
                        x_col = tool_arguments.get("x_col")
                        y_col = tool_arguments.get("y_col")
                        color_col = tool_arguments.get("color_col")
                        size_col = tool_arguments.get("size_col")
                        title = tool_arguments.get("title", "My Chart")
                        width = tool_arguments.get("width", 800)
                        height = tool_arguments.get("height", 600)
                        orientation = tool_arguments.get("orientation", "v")
                        labels = tool_arguments.get("labels", {})
                        template = tool_arguments.get("template", "plotly_white")

                        # Call a hypothetical method that returns Plotly JSON
                        create_graph_json = self.create_graph(
                            query=query,
                            chart_type=chart_type,
                            x_col=x_col,
                            y_col=y_col,
                            color_col=color_col,
                            size_col=size_col,
                            title=title,
                            width=width,
                            height=height,
                            orientation=orientation,
                            labels=labels,
                            template=template
                        )

                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": create_graph_json
                        })

                    # If this is the last tool call, check if we need to return
                    if idx == len(tool_calls) - 1:
                        # If the LLM wants more steps, the loop will continue
                        # Otherwise, we break once no more tool calls are made
                        pass

            else:
                # No tool calls => direct text answer
                response_text = assistant_message.content
                break

        return {
            "response": response_text,
            "executed_queries": list_of_executed_queries
        }