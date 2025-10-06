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
| 1 | **[Mohs Hardness Dataset](https://www.kaggle.com/competitions/playground-series-s3e25)** | Regression | <a href="1. Regression with a Mohs Hardness Dataset/Regression with a Mohs Hardness Dataset.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a> <a href="https://huggingface.co/spaces/etuncer/mohs-hardness-prediction-kaggle"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a> |
| 2 | **[Predicting Road Accident Risk](https://www.kaggle.com/competitions/playground-series-s5e10)** | Regression | <a href="2. Predicting Road Accident Risk/Predicting Road Accident Risk.ipynb"><img alt="Notebook" src="https://img.shields.io/badge/Notebook-f37726?style=for-the-badge&logo=jupyter&logoColor=white"></a> <a href="https://huggingface.co/spaces/etuncer/road-accident-risk-kaggle"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"></a> |

---

### Contact
For questions or suggestions, please open an issue or reach out to my social accounts via my GitHub profile.

This README will evolve as new competition solutions are added.