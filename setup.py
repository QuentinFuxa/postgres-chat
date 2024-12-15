from setuptools import setup, find_packages

setup(
    name='rag-handler',
    version='0.1.0',
    description='Retrieval-Augmented Generation Handler using PostgreSQL and OpenAI',
    author='Quentin Fuxa',
    packages=find_packages(),
    install_requires=[
        'openai',
        'pandas',
        'psycopg2',
        'sqlalchemy',
        'plotly'
    ],
    python_requires='>=3.9'
)