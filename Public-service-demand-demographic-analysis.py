
# ============================================================
# PUBLIC SERVICE DEMAND AND DEMOGRAPHIC ANALYSIS
# ============================================================
#
# Project Overview:
# This project uses Python and pandas to analyze public service
# assistance records from 2025 and 2026.
#
# The analysis focuses on identifying the most frequently used
# programs by year, the most requested program in each city, and
# the average age of people served by each program.
#
# Objectives:
# 1. Identify the most frequently used program each year.
# 2. Determine the most requested program in each city.
# 3. Calculate the average age of people served by each program.
#
# Technologies: Python 3, pandas, Jupyter Notebook
# Skills: Data Cleaning, Data Analysis, Data Aggregation,
# Date Processing, Demographic Analysis, Business Intelligence
#
# Dataset:
# Informe Atenciones.csv
#
# Note: The dataset must be available in the working directory.
# ============================================================


# 1. IMPORT LIBRARIES AND LOAD THE DATASET
# ============================================================

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# Load the dataset using a semicolon as the delimiter
file_path = "Informe Atenciones.csv"

df = pd.read_csv(file_path, sep=";", dtype=str)

print("=" * 70)
print("PUBLIC SERVICE DEMAND AND DEMOGRAPHIC ANALYSIS")
print("=" * 70)

print("\nDataset Overview")
print("-" * 70)
print(f"Number of records: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumn Information:")
df.info()


# 2. DATA PREPARATION AND CLEANING
# ============================================================

# Remove leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Remove unnecessary spaces from text fields
text_columns = df.select_dtypes(include=["object"]).columns

for column in text_columns:
    df[column] = df[column].str.strip()

# Convert registration dates to datetime format
df["FECHA REGISTRO"] = pd.to_datetime(
    df["FECHA REGISTRO"],
    dayfirst=True,
    errors="coerce"
)

# Convert age values to numeric format
df["EDAD"] = pd.to_numeric(df["EDAD"], errors="coerce")

# Extract the year from the registration date
df["YEAR"] = df["FECHA REGISTRO"].dt.year

# Display missing values for relevant analysis fields
print("\nData Quality Check")
print("-" * 70)

relevant_columns = ["FECHA REGISTRO", "CIUDAD", "PROGRAMA", "EDAD"]

print(df[relevant_columns].isna().sum().rename("Missing Values"))


# 3. MOST FREQUENTLY USED PROGRAM BY YEAR
# ============================================================

print("\n\n3. MOST FREQUENTLY USED PROGRAM BY YEAR")
print("=" * 70)

# Count records for each program and year
yearly_program_usage = (
    df.dropna(subset=["YEAR", "PROGRAMA"])
      .groupby(["YEAR", "PROGRAMA"])
      .size()
      .reset_index(name="TOTAL RECORDS")
      .sort_values(["YEAR", "TOTAL RECORDS"],
                   ascending=[True, False])
)

print("\nProgram Usage by Year:")
print(yearly_program_usage.to_string(index=False))

# Identify the most frequently used program in each year
top_program_by_year = (
    yearly_program_usage
    .groupby("YEAR", group_keys=False)
    .head(1)
)

print("\nMost Frequently Used Program in Each Year:")
print(top_program_by_year.to_string(index=False))

print(
    "\nInterpretation: The results identify the program with the "
    "highest number of recorded services in each year."
)


# 4. MOST REQUESTED PROGRAM BY CITY
# ============================================================

print("\n\n4. MOST REQUESTED PROGRAM BY CITY")
print("=" * 70)

# Count records by city and program
city_program_demand = (
    df.dropna(subset=["CIUDAD", "PROGRAMA"])
      .groupby(["CIUDAD", "PROGRAMA"])
      .size()
      .reset_index(name="TOTAL REQUESTS")
)

# Identify the program with the highest demand in each city.
# If multiple programs share the maximum, all tied programs
# are retained.
city_maximums = city_program_demand.groupby(
    "CIUDAD"
)["TOTAL REQUESTS"].transform("max")

top_program_by_city = (
    city_program_demand[
        city_program_demand["TOTAL REQUESTS"] == city_maximums
    ]
    .sort_values(["CIUDAD", "PROGRAMA"])
    .reset_index(drop=True)
)

print("\nMost Requested Program in Each City:")
print(top_program_by_city.to_string(index=False))

print(
    "\nInterpretation: This table shows the program with the "
    "highest recorded demand in each city. Ties are included."
)


# 5. AVERAGE AGE BY PROGRAM
# ============================================================

print("\n\n5. AVERAGE AGE BY PROGRAM")
print("=" * 70)

# Calculate the average age for each program.
# Invalid or missing age values are excluded automatically.
average_age_by_program = (
    df.dropna(subset=["PROGRAMA", "EDAD"])
      .groupby("PROGRAMA")["EDAD"]
      .agg(["mean", "count"])
      .reset_index()
      .rename(columns={
          "mean": "AVERAGE AGE",
          "count": "VALID AGE RECORDS"
      })
      .sort_values("AVERAGE AGE", ascending=False)
)

# Round average ages to two decimal places
average_age_by_program["AVERAGE AGE"] = (
    average_age_by_program["AVERAGE AGE"].round(2)
)

print("\nAverage Age by Program:")
print(average_age_by_program.to_string(index=False))

print(
    "\nInterpretation: This table compares the average age of "
    "people served by each program. The valid record count helps "
    "show how many age values support each calculation."
)


# 6. FINAL SUMMARY
# ============================================================

print("\n\n6. FINAL SUMMARY")
print("=" * 70)

print("\nRecords by Year:")
print(
    df["YEAR"]
    .value_counts()
    .sort_index()
    .rename_axis("YEAR")
    .to_string()
)

print("\nTop Program by Year:")
for _, row in top_program_by_year.iterrows():
    print(
        f"{int(row['YEAR'])}: {row['PROGRAMA']} "
        f"({row['TOTAL RECORDS']:,} records)"
    )

print("\nProgram with the Highest Demand Overall:")
overall_program_demand = (
    df["PROGRAMA"]
    .dropna()
    .value_counts()
)

if not overall_program_demand.empty:
    print(
        f"{overall_program_demand.index[0]} "
        f"({overall_program_demand.iloc[0]:,} records)"
    )
else:
    print("No valid program records were available.")

print("\nAnalysis completed successfully.")
print("=" * 70)


# 7. EXPORT RESULTS
# ============================================================

# Save analysis results as CSV files for further reporting
yearly_program_usage.to_csv(
    "program_usage_by_year.csv",
    index=False
)

top_program_by_city.to_csv(
    "top_program_by_city.csv",
    index=False
)

average_age_by_program.to_csv(
    "average_age_by_program.csv",
    index=False
)

print("\nExported Files:")
print("- program_usage_by_year.csv")
print("- top_program_by_city.csv")
print("- average_age_by_program.csv")
