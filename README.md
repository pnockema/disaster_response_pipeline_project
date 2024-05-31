# Disaster Response Pipeline Project

### Summary
This disaster response pipeline's purpose is to support organisations that are involved in providing help in disastrous events. The pipeline consists of an ETL pipeline and a machine learning model that. The ETL (Extract-Transform-Load) pipeline takes in a dataset of text messages which are classified into disaster relevant categories. It outputs a dataset that can be used to train an ML (Machine Learning) model to categorize further text messages automatically. Thus, governments and NGOs providing help in disaster events can quickly and model driven classify disaster text messages into various categories of needs which ought to make coordination of help easier.

### Files in the repository
#### File structure:
- disaster_responser_pipeline_project
    - app
        - run.py
        - templates
            - go.html
            - master.html
    - data
        - disaster_categories.csv
        - disaster_messages.csv
        - disaster.db
        - DisasterResponse.db
        - ETL Pipeline Preparation.ipynb
        - ML Pipeline Preparation.ipynb
        - process_data.py
    - models
        - train_classifier.py
    - README.md

#### File descriptions:
- run.py, go.html, master.html:
    - These files setup the layout and visualisations of the web app
- disaster_categories.csv, disaster_messages.csv:
    - Data files containing disaster messages and corresponding categories
- disaster.db:
    - test-.db-file containing prepared data after ETL process from Jupyter Notebook
- DisasterResponse.db:
    - db-file containing prepared data after ETL process from process_data.py
- ETL Pipeline Preparation.ipynb, ML Pipeline Preparation.ipynb:
    - Jupyter Notebooks containing my personal exploration on ETL and ML process
- process_data.py:
    - Python file containing functions that set up ETL pipeline
- train_classifier.py:
    - Python file containing functions for NLP and ML pipelines

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to the provided link 