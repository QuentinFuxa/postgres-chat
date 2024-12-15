from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_desc = f.read()

setup(
    name='postgres-chat',
    version='0.1.3',
    description='Retrieval-Augmented Generation Handler using PostgreSQL and OpenAI',
    long_description=long_desc,
    long_description_content_type='text/markdown',
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
