# Kaggle Demo: RFM Analysis in 5 MCP Prompts

Replicates kadirduran's [RFM Analysis, K-Means Clustering & Cohort Analysis](https://www.kaggle.com/code/kadirduran/rfm-analysis-k-means-clustering-cohort-analysis) using LocalData MCP tools instead of manual pandas/sklearn code.

## Dataset

UCI Online Retail dataset (~541K transactions, UK retailer, 2010-2011).

Download:
```bash
curl -L -o online_retail.zip "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
unzip online_retail.zip
python3 -c "import pandas as pd; df = pd.read_excel('Online Retail.xlsx'); df.to_csv('online_retail.csv', index=False)"
```

## What the demo covers

| Step | Traditional (pandas/sklearn) | MCP Tool |
|------|------------------------------|----------|
| EDA | `df.describe()`, null checks, dtypes | `get_data_quality_report` |
| Cohort analysis | 30+ lines of groupby/pivot | `analyze_time_series` |
| RFM segmentation | 40+ lines of scoring | `analyze_rfm` |
| K-means clustering | Elbow method, silhouette, fit | `analyze_clusters` |
| Business summary | Manual aggregation | Combined tool outputs |
| Extension: anomaly detection | Not in original | `detect_anomalies` + `analyze_hypothesis_test` |

## Running the notebook

Open `online_retail_mcp_demo.ipynb` in Jupyter, or use the Colab version:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChrisGVE/localdata-mcp/blob/main/examples/kaggle-demo/online_retail_mcp_demo_colab.ipynb)
