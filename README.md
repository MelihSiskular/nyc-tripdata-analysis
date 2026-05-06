# NYC Trip Data Analysis (Uber/Lyft)

This project analyzes 22M+ High Volume For-Hire Vehicle (HVFHV) trip records in New York City for March 2026.

The main goal is to understand ride demand, pricing behavior, and driver earnings using real-world Uber/Lyft-style data.

---

# Dataset

The dataset is provided by NYC Taxi & Limousine Commission (TLC):

https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

There are different types of datasets (Yellow Taxi, Green Taxi, FHV, HVFHV).  
In this project, I focus on **High Volume FHV (Uber/Lyft)** data.

---

# Data Structure

Each row represents a single trip.

Key fields include:
- request_datetime → when the ride was requested
- pickup_datetime → when the ride started
- dropoff_datetime → when the ride ended
- trip_miles → distance of the trip
- trip_time → duration of the trip (seconds)
- base_passenger_fare → fare paid by the passenger
- driver_pay → earnings of the driver

<img src="https://github.com/user-attachments/assets/d94c06fe-e153-408a-bc1d-04f7f0499d40" width="400"/>

---

# Location Information

NYC consists of 5 boroughs:
- Manhattan
- Brooklyn
- Queens
- Bronx
- Staten Island

Instead of coordinates, the dataset uses **Taxi Zone IDs**.

Example:
- Central Park → Zone 43


<img src="https://github.com/user-attachments/assets/e8a4fcaf-8bd6-4971-b942-8b7262084dcd" width="400"/>

https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv with this link you can see each id & zone pairs.

---

# Working with Parquet Format

The dataset is stored in **Parquet format**, which is optimized for big data:

- Smaller file size
- Faster read times
- Column-based structure

---

# Setup

Install required libraries:

```bash
pip install pandas
pip install matplotlib
pip install numpy
```

Import these libraries:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
```
---

# Configuration
To display dataset properly:

```python
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
```

Correct file path:
```python
FILE_PATH = "fhvhv_tripdata_2026-03.parquet"
```

---

# Load Data

```python
cols = [ ... ]  # selected columns

df = pd.read_parquet(FILE_PATH, columns=cols)

print("Raw rows:", len(df))
```
```md
Raw rows: 22,058,358
```

---

# Data Cleaning

The dataset may contain unreliable or invalid records, so basic data cleaning is required.

```python
df_clean = df[
    (df["trip_miles"] > 0) &
    (df["trip_time"] > 0) &
    (df["base_passenger_fare"] > 0) &
    (df["driver_pay"] >= 0)
].copy()

print("New rows:", len(df_clean))
```

```md
New rows: 22,047,566
```

After applying basic filtering rules, a small portion of the dataset (around 10K rows) was removed.  
This ensures that only valid and realistic trips are used for further analysis.

---

# Feature Engineering

To extract meaningful insights, several new features were created from the raw dataset:

- pickup_hour → hour of the trip
- trip_duration_min → trip duration in minutes
- speed_mph → average trip speed
- wait_time_min → passenger waiting time
- price_per_mile → fare efficiency metric
- driver_share → driver earning ratio

These features allow deeper analysis for our goal.

```python
# Data & Time Features
df_clean["pickup_hour"] = df_clean["pickup_datetime"].dt.hour
df_clean["pickup_weekday"] = df_clean["pickup_datetime"].dt.weekday
df_clean["is_weekend"] = df_clean["pickup_weekday"] >= 5
```

```python
# Operational Features
df_clean["trip_duration_min"] = df_clean["trip_time"] / 60
df_clean["speed_mph"] = df_clean["trip_miles"] / (df_clean["trip_time"] / 3600)

df_clean["wait_time_min"] = (
    df_clean["pickup_datetime"] - df_clean["request_datetime"]
).dt.total_seconds() / 60

df_clean["driver_arrival_min"] = (
    df_clean["on_scene_datetime"] - df_clean["request_datetime"]
).dt.total_seconds() / 60
```

```python
# Price Features
df_clean["passenger_total"] = (
    df_clean["base_passenger_fare"] +
    df_clean["tolls"] +
    df_clean["bcf"] +
    df_clean["sales_tax"] +
    df_clean["congestion_surcharge"] +
    df_clean["airport_fee"] +
    df_clean["tips"] +
    df_clean["cbd_congestion_fee"]
)

df_clean["price_per_mile"] = df_clean["base_passenger_fare"] / df_clean["trip_miles"]
df_clean["driver_pay_per_mile"] = df_clean["driver_pay"] / df_clean["trip_miles"]
df_clean["driver_share"] = df_clean["driver_pay"] / df_clean["base_passenger_fare"]

```

---

# Final Data Filtering

After feature engineering, I applied additional filtering rules to remove unrealistic values created from the new features.

```python
before_final_filter = len(df_clean)

df_clean = df_clean[
    (df_clean["speed_mph"] > 1) &
    (df_clean["speed_mph"] < 80) &
    (df_clean["wait_time_min"] >= 0) &
    (df_clean["wait_time_min"] < 120) &
    (df_clean["trip_duration_min"] < 180)
].copy()

after_final_filter = len(df_clean)

print("Rows before final filtering:", before_final_filter)
print("Rows after final filtering:", after_final_filter)
print("Removed rows:", before_final_filter - after_final_filter)
```

```md
Rows before final filtering: 22047566
Rows after final filtering: 21748865
Removed rows: 298701
```

After feature engineering, a final filtering step was applied to remove unrealistic values such as extreme speeds, abnormal waiting times, and excessively long trips.  

We have better working environment now.

---

# Exploratory Data Analysis (EDA)

After cleaning and feature engineering, I analyzed the dataset and I focused on:

- At what hours is ride demand highest?
- How does passenger fare change by hour?
- When do passengers wait longer?
- How much of the passenger fare goes to the driver?
- Are there signs of surge pricing?

First, I define a directory to store all generated figures:

```python
FIGURE_DIR = "reports/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)
```

All generated plots will be saved in the reports/figures folder.

---

## 1. Hourly Trip Demand

```python
hourly_trips = df_clean.groupby("pickup_hour").size()

plt.figure(figsize=(12, 5))
plt.style.use("seaborn-v0_8")
plt.plot(hourly_trips.index, hourly_trips.values, marker="o")
plt.title("Hourly HVFHV Trip Count - March 2026")
plt.xlabel("Hour")
plt.ylabel("Trip Count")
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/hourly_trip_count.png", dpi=300)
plt.show()
```

<img width="3600" height="1500" alt="hourly_trip_count" src="https://github.com/user-attachments/assets/d2233998-9e67-4ad2-90d4-6638c16f1733" />

**Insight:**

Ride demand is lowest between 2 AM and 5 AM, then increases sharply during morning hours.  
Demand peaks in the evening (around 5–7 PM), likely due to social activities.

---

## 2. Average Fare By Hour

```python
hourly_fare = df_clean.groupby("pickup_hour")["base_passenger_fare"].mean()

plt.figure(figsize=(12, 5))
plt.plot(hourly_fare.index, hourly_fare.values, marker="o")
plt.title("Average Base Passenger Fare by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Fare ($)")
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/avg_fare_by_hour.png", dpi=300)
plt.show()

```
<img width="3600" height="1500" alt="avg_fare_by_hour" src="https://github.com/user-attachments/assets/7872c005-c546-42ae-9665-9486992f1587" />

**Insight:**

Average fares are higher during early morning hours when demand is low, suggesting possible surge pricing due to limited driver availability.  
Prices stabilize during the day and slightly increase again in the evening.

---

## 3. Avergae Wait Time

```python
hourly_wait = df_clean.groupby("pickup_hour")["wait_time_min"].mean()

plt.figure(figsize=(12, 5))
plt.plot(hourly_wait.index, hourly_wait.values, marker="o")
plt.title("Average Wait Time by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Wait Time (min)")
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/avg_wait_time_by_hour.png", dpi=300)
plt.show()
```

<img width="3600" height="1500" alt="avg_wait_time_by_hour" src="https://github.com/user-attachments/assets/aa6a96fd-e32f-4528-afa9-79ac39a7d036" />

**Insight:**

Passenger wait time is highest during early morning hours and gradually decreases during the day.  
This indicates lower driver availability at night and improved service during peak hours.

---

## 4. Driver Share

```python
hourly_driver_share = df_clean.groupby("pickup_hour")["driver_share"].mean()

plt.figure(figsize=(12, 5))
plt.plot(hourly_driver_share.index, hourly_driver_share.values, marker="o")
plt.title("Average Driver Share by Hour")
plt.xlabel("Hour")
plt.ylabel("Driver Share")
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/driver_share_by_hour.png", dpi=300)
plt.show()
```

<img width="3600" height="1500" alt="driver_share_by_hour" src="https://github.com/user-attachments/assets/99032db7-ffd5-4923-a0e5-614b714f5c3f" />

**Insight:**

Driver share slightly increases during peak hours, reaching its highest levels in the evening.  
This suggests that drivers may benefit more during high-demand periods.

---

## 5. Trip Distance vs Fare

```python
# Due to the large size of the dataset, a random sample of 300,000 rows was used for visualization.

sample_df = df_clean.sample(300_000, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(
    sample_df["trip_miles"],
    sample_df["base_passenger_fare"],
    alpha=0.08,
    s= 10
)
plt.title("Trip Distance vs Base Passenger Fare")
plt.xlabel("Trip Miles")
plt.ylabel("Base Passenger Fare ($)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/trip_distance_vs_fare.png", dpi=300)
plt.show()
```

<img width="2400" height="1800" alt="trip_distance_vs_fare" src="https://github.com/user-attachments/assets/a99f3ee3-abec-459e-8f18-ac5c500e629e" />



**Insight:**

There is a clear positive relationship between trip distance and fare.  
However, the spread increases for longer trips, indicating pricing variability.

---


# Airport Trip Analysis

As a final real-world case study, I analyzed trips related to NYC airports.

Airport trips were separated into two groups:

- **From Airport:** trips starting from an airport zone
- **To Airport:** trips ending at an airport zone

The airport zones used in this analysis:

- Newark Airport → Zone 1
- JFK Airport → Zone 132
- LaGuardia Airport → Zone 138


```python
# Airport zone IDs
AIRPORT_ZONES = {
    1: "Newark Airport",
    132: "JFK Airport",
    138: "LaGuardia Airport"
}

airport_zone_ids = list(AIRPORT_ZONES.keys())
```

```python
# Trips from airport
from_airport = df_clean[df_clean["PULocationID"].isin(airport_zone_ids)].copy()
from_airport["airport"] = from_airport["PULocationID"].map(AIRPORT_ZONES)
from_airport["trip_type"] = "From Airport"
```
```python
# Trips to airport
to_airport = df_clean[df_clean["DOLocationID"].isin(airport_zone_ids)].copy()
to_airport["airport"] = to_airport["DOLocationID"].map(AIRPORT_ZONES)
to_airport["trip_type"] = "To Airport"
```

```python
# Combine both
airport_trips = pd.concat([from_airport, to_airport], ignore_index=True)

print("Airport trip rows:", len(airport_trips))
airport_trips.head()
```

**Summary**

```python
print("Airport trip rows:", len(airport_trips))

airport_trips.head()
```

```md
Airport trip rows: 1652439

             airport     trip_type  trip_count  avg_distance  avg_duration_min  avg_fare  avg_driver_pay  avg_wait_time  avg_price_per_mile
0        JFK Airport  From Airport      324516         18.00             45.17     81.92           59.20           7.40                4.79
1        JFK Airport    To Airport      366414         14.08             41.19     72.11           49.43           5.66                5.19
2  LaGuardia Airport  From Airport      406887         11.61             33.80     61.12           43.34           6.89                5.68
3  LaGuardia Airport    To Airport      397906          9.41             27.96     52.35           34.18           5.06                5.66
4     Newark Airport    To Airport      156716         17.01             41.37     92.17           60.35           4.56                5.51

```

**Note:**

Trips originating from Newark Airport are not present in this dataset.

This is likely because Newark Airport is located in New Jersey, while the TLC dataset primarily covers trips within New York City.

As a result, trips going **to Newark Airport** are included, but trips starting **from Newark Airport** are not captured in this dataset.

---

## Airport Visulazation

```python
# Airport Summary
airport_summary = airport_trips.groupby(["airport", "trip_type"]).agg(
    trip_count=("trip_miles", "count"),
    avg_distance=("trip_miles", "mean"),
    avg_duration_min=("trip_duration_min", "mean"),
    avg_fare=("base_passenger_fare", "mean"),
    avg_driver_pay=("driver_pay", "mean"),
    avg_wait_time=("wait_time_min", "mean"),
    avg_price_per_mile=("price_per_mile", "mean")
).reset_index()

airport_summary = airport_summary.round(2)
```

## 1 - Airport Average Fare
```python
airport_fare = airport_summary.pivot(
    index="airport",
    columns="trip_type",
    values="avg_fare"
)

plt.figure(figsize=(10, 5))
airport_fare.plot(kind="bar", figsize=(10, 5))
plt.title("Average Fare: To Airport vs From Airport")
plt.xlabel("Airport")
plt.ylabel("Average Base Fare ($)")
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/airport_avg_fare.png", dpi=300)
plt.show()
```

<img width="3000" height="1500" alt="airport_avg_fare" src="https://github.com/user-attachments/assets/d40a37b3-f86e-45b2-9f9f-5aa5589f63fc" />

**Insight**

Trips to Newark Airport have the highest average fare, likely due to its longer distance from Manhattan and cross-state travel.

---

## 2 - Airport Average Duration

```python
airport_duration = airport_summary.pivot(
    index="airport",
    columns="trip_type",
    values="avg_duration_min"
)

plt.figure(figsize=(10, 5))
airport_duration.plot(kind="bar", figsize=(10, 5))
plt.title("Average Trip Duration: To Airport vs From Airport")
plt.xlabel("Airport")
plt.ylabel("Average Duration (min)")
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/airport_avg_duration.png", dpi=300)
plt.show()
```

<img width="3000" height="1500" alt="airport_avg_duration" src="https://github.com/user-attachments/assets/f1395074-7d65-4bd0-8070-58ef0b576626" />

**Insight**

JFK trips have the longest duration, reflecting its distance and heavy traffic conditions compared to LaGuardia.

---

## 3 - Airport Average Waiting

```python
airport_wait = airport_summary.pivot(
    index="airport",
    columns="trip_type",
    values="avg_wait_time"
)

plt.figure(figsize=(10, 5))
airport_wait.plot(kind="bar", figsize=(10, 5))
plt.title("Average Wait Time: To Airport vs From Airport")
plt.xlabel("Airport")
plt.ylabel("Average Wait Time (min)")
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/airport_avg_wait_time.png", dpi=300)
plt.show()
```

<img width="3000" height="1500" alt="airport_avg_wait_time" src="https://github.com/user-attachments/assets/3bff4d54-d79a-43fa-9089-389566a2236e" />

**Insight**

Passengers experience the longest wait times when leaving JFK Airport, suggesting high demand and limited driver availability in that zone.

---

# Radar Chart: Popular Pickup vs Dropoff Zones

To compare the most active zones, I created a radar chart using the top pickup and dropoff locations.

Since raw trip counts are very large, values were normalized between 0 and 1.  
This makes it easier to compare pickup and dropoff activity patterns visually.

```python
# Load File
zone_lookup = pd.read_csv("taxi_zone_lookup.csv")
```

```python
# Combine pickup and dropoff zones
all_zone_counts = (
    pd.concat([
        df_clean["PULocationID"],
        df_clean["DOLocationID"]
    ])
    .value_counts()
    .reset_index()
)

all_zone_counts.columns = ["LocationID", "total_count"]
```

```python
# Add zone names
all_zone_counts = all_zone_counts.merge(
    zone_lookup[["LocationID", "Zone"]],
    on="LocationID",
    how="left"
)

# Remove Outside of NYC
all_zone_counts = all_zone_counts[all_zone_counts["Zone"] != "Outside of NYC"]
```

``` python
# Select top 8 most popular zones
top_zones = all_zone_counts.head(8)["LocationID"]
```

```python
# Pickup counts for selected zones
pickup_counts = (
    df_clean[df_clean["PULocationID"].isin(top_zones)]
    .groupby("PULocationID")
    .size()
)

# Dropoff counts for selected zones
dropoff_counts = (
    df_clean[df_clean["DOLocationID"].isin(top_zones)]
    .groupby("DOLocationID")
    .size()
)
```

```python
# Create radar dataframe
radar_df = pd.DataFrame({
    "LocationID": top_zones
})

radar_df["pickup"] = radar_df["LocationID"].map(pickup_counts).fillna(0)
radar_df["dropoff"] = radar_df["LocationID"].map(dropoff_counts).fillna(0)
```

```python
# Zone names
radar_df = radar_df.merge(zone_lookup, on="LocationID")
```

```python
# Add zone names
radar_df = radar_df.merge(
    zone_lookup[["LocationID", "Zone"]],
    on="LocationID",
    how="left"
)
```

```python
# Sort by total activity
radar_df["total"] = radar_df["pickup"] + radar_df["dropoff"]
radar_df = radar_df.sort_values("total", ascending=False)
```

```python
# Normalize values between 0 and 1
radar_df["pickup_norm"] = radar_df["pickup"] / radar_df["pickup"].max()
radar_df["dropoff_norm"] = radar_df["dropoff"] / radar_df["dropoff"].max()

labels = radar_df["Zone_short"].tolist()
pickup_values = radar_df["pickup_norm"].tolist()
dropoff_values = radar_df["dropoff_norm"].tolist()
```

```python
# Radar chart
labels += labels[:1]
pickup_values += pickup_values[:1]
dropoff_values += dropoff_values[:1]

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

plt.style.use("seaborn-v0_8")

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

ax.plot(angles, pickup_values, linewidth=2, label="Pickup Zones")
ax.fill(angles, pickup_values, alpha=0.25)

ax.plot(angles, dropoff_values, linewidth=2, label="Dropoff Zones")
ax.fill(angles, dropoff_values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels[:-1], fontsize=9)

ax.set_title(
    "Most Popular NYC Uber/Lyft Zones: Pickup vs Dropoff",
    fontsize=15,
    pad=30
)

ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/radar_top_zones_pickup_dropoff.png", dpi=300)
plt.show()

radar_df[["Zone", "pickup", "dropoff", "total"]]
```

<img width="3000" height="3000" alt="radar_top_zones_pickup_dropoff" src="https://github.com/user-attachments/assets/f0857195-b877-4a8f-bf37-1e819a096f4e" />

### Insight

Airport zones dominate both pickup and dropoff activity, especially LaGuardia and JFK.

Interestingly, pickup activity is generally higher than dropoff in central zones, suggesting that users tend to start their trips in busy urban areas and end them in more distributed locations.

---

# Conclusion

In this project, I analyzed over 22 million Uber/Lyft trips in New York City using High Volume FHV data.

Starting from raw parquet files, I performed data cleaning, feature engineering, and exploratory data analysis to uncover real-world mobility patterns.

---

# Author

Melih Şişkular  
