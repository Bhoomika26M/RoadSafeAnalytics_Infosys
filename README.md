# RoadSafe Analytics

### Road Accidents – Exploratory Data Analysis (EDA)

##  Project Overview

**RoadSafe Analytics** is an Exploratory Data Analysis (EDA) project focused on understanding patterns, trends, and contributing factors behind road accidents using a large real-world dataset.
The analysis aims to derive actionable insights that can support **road safety awareness, decision-making, and policy recommendations**.

This project uses Python-based data analysis and visualization techniques to explore accident severity, time-based patterns, weather conditions, and geospatial trends.

---

##  Project Objectives

* Analyze a large-scale road accident dataset to identify meaningful patterns
* Understand factors influencing accident severity
* Perform comprehensive EDA using visual and statistical techniques
* Extract insights related to time, weather, visibility, and location
* Present findings clearly through documentation and visualizations


##  How to Run the Project Locally

### 1️ Clone the Repository

```bash
git clone https://github.com/your-username/RoadSafe-Analytics.git
cd RoadSafe-Analytics
```

### 2️ Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate
```

### 3️ Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4️ Download the Dataset

1. Visit the Kaggle dataset link
2. Download `US_Accidents.csv`
3. Place it inside:

```
data/raw/US_Accidents.csv
```

### 5️ Run Data Preprocessing

```bash
python scripts/data_preprocessing.py
```

### 6️ Run Analysis Notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open notebooks in sequence from the `notebooks/` folder.


