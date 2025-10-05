# 20 Kaggle Competition Solutions

A practical portfolio of 20 end-to-end solutions for various Kaggle competitions. This repository documents my journey of systematically dissecting each challenge, from initial data exploration and studying top-ranking approaches to developing and deploying my own predictive models.

### What you’ll find in each competition folder
-   **Problem Overview**: A clear summary of the competition's objective, dataset, and evaluation metric.
-   **Exploratory Data Analysis (EDA)**: A detailed notebook investigating the data, uncovering patterns, and generating insights.
-   **Analysis of Top Solutions**: A summary of strategies and techniques used by high-ranking participants to inform my own approach.
-   **Modeling Notebook**: The complete workflow, including feature engineering, model selection, training, and evaluation.
-   **Interactive Demo**: A live Streamlit application deployed on Hugging Face Spaces to demonstrate the model's functionality (when applicable).

### Getting Started (Poetry)
1.  **Install Poetry** (if not installed)
    -   `curl -sSL https://install.python-poetry.org | python3 -`
    -   Ensure Poetry is on your PATH (restart shell or follow installer notes).
2.  **Clone the repository**
    -   `git clone https://github.com/emretuncer256/20-kaggle-competitions.git`
    -   `cd 20-kaggle-competitions`
3.  **Install shared dependencies** at the repository root
    -   `poetry install`
    -   Optional: `poetry shell` to enter the virtual environment.

### How to run
-   **Notebooks**: Open the `.ipynb` files directly or launch Jupyter with Poetry from the repo root:
    -   `poetry run jupyter lab`
-   **Streamlit apps**: From the specific project folder, run:
    -   `poetry run streamlit run app.py`

### Tech Stack
-   **Core ML**: Python, pandas, NumPy, scikit-learn
-   **Boosting Models**: XGBoost, LightGBM
-   **Visualization**: Matplotlib, Seaborn, Plotly
-   **Apps**: Streamlit, Hugging Face Spaces
-   **Environment**: Poetry, Jupyter

---

## Competition Solutions

Here is a list of the competitions I have solved. Each entry includes links to the official Kaggle page, my solution notebook, and a live demo.

| # | Competition Name | Problem Type | Links |
|:-:|:---|:---|:---|

---

### Contact
For questions or suggestions, please open an issue or reach out to my social accounts via my GitHub profile.

This README will evolve as new competition solutions are added.