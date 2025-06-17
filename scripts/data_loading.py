import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("data/MachineLearningRating_v3.txt", sep="|")

# print("\nFirst 5 Rows:\n")
# print(df.head())

# print("\nInfo:\n")
# # print(df.info())

# print("\nSummary Statistics:\n")
# # print(df.describe())

# print("\nTotal Number of data points:", len(df))

# missing_pct = (df.isnull().sum() / len(df)) * 100
# cols_to_drop = missing_pct[missing_pct > 90].index

# df_cleaned = df.drop(columns=cols_to_drop)

# print("\nNumber of Missing Values in Each Column After Dropping Columns with > 90% Missing Values:\n")
# print(df_cleaned.isnull().sum())

# df_missing_pct = (df_cleaned.isnull().sum() / len(df_cleaned)) * 100
# cols_with_missing = df_missing_pct[df_missing_pct > 0].index

# print("\nColumns with Missing Values:\n")
# print(cols_with_missing)

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("report/profile_report.html")

