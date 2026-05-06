import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

FILE_PATH = "data/fhvhv_tripdata_2026-03.parquet"

# Required Colons
cols = [
    "hvfhs_license_num",
    "dispatching_base_num",
    "originating_base_num",
    "request_datetime",
    "on_scene_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
    "cbd_congestion_fee"
]

df = pd.read_parquet(FILE_PATH, columns=cols)

print("Raw rows:", len(df))
print(df.head())
print(df.info())

before = len(df)

# 2) Data Cleaning
df_clean = df[
    (df["trip_miles"] > 0) &
    (df["trip_time"] > 0) &
    (df["base_passenger_fare"] > 0) &
    (df["driver_pay"] >= 0)
].copy()

print(len(df_clean))

# 3) Data & Time Features
df_clean["pickup_hour"] = df_clean["pickup_datetime"].dt.hour
df_clean["pickup_day"] = df_clean["pickup_datetime"].dt.day
df_clean["pickup_weekday"] = df_clean["pickup_datetime"].dt.day_name()
df_clean["is_weekend"] = df_clean["pickup_datetime"].dt.weekday >= 5

# 4) Operational Features
df_clean["trip_duration_min"] = df_clean["trip_time"] / 60
df_clean["speed_mph"] = df_clean["trip_miles"] / (df_clean["trip_time"] / 3600)

df_clean["wait_time_min"] = (
    df_clean["pickup_datetime"] - df_clean["request_datetime"]
).dt.total_seconds() / 60

df_clean["driver_arrival_min"] = (
    df_clean["on_scene_datetime"] - df_clean["request_datetime"]
).dt.total_seconds() / 60

# 5) Price Features
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

# 6) New data
df_clean = df_clean[
    (df_clean["speed_mph"] > 1) &
    (df_clean["speed_mph"] < 80) &
    (df_clean["wait_time_min"] >= 0) &
    (df_clean["wait_time_min"] < 120) &
    (df_clean["trip_duration_min"] < 180)
]

after = len(df_clean)

print("Before:", before)
print("After:", after)
print("Removed:", before - after)
print("Removed %:", round((before - after) / before * 100, 2))

print(df_clean.describe())

print("Average trip miles:", df_clean["trip_miles"].mean())
print("Average trip duration min:", df_clean["trip_duration_min"].mean())
print("Average fare:", df_clean["base_passenger_fare"].mean())
print("Average driver pay:", df_clean["driver_pay"].mean())
print("Average wait time min:", df_clean["wait_time_min"].mean())
print("Average speed mph:", df_clean["speed_mph"].mean())



FIGURE_DIR = "reports/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)


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

sample_df = df_clean.sample(300_000, random_state=42)

plt.figure(figsize=(10, 5))
plt.hist(sample_df["price_per_mile"], bins=80)
plt.title("Price per Mile Distribution")
plt.xlabel("Price per Mile ($)")
plt.ylabel("Trip Count")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/price_per_mile_distribution.png", dpi=300)
plt.show()

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

summary_metrics = {
    "Total Clean Trips": len(df_clean),
    "Average Trip Distance (miles)": round(df_clean["trip_miles"].mean(), 2),
    "Average Trip Duration (min)": round(df_clean["trip_duration_min"].mean(), 2),
    "Average Base Fare ($)": round(df_clean["base_passenger_fare"].mean(), 2),
    "Average Driver Pay ($)": round(df_clean["driver_pay"].mean(), 2),
    "Average Wait Time (min)": round(df_clean["wait_time_min"].mean(), 2),
    "Average Speed (mph)": round(df_clean["speed_mph"].mean(), 2),
    "Average Driver Share": round(df_clean["driver_share"].mean(), 2)
}


# Airport zone IDs
AIRPORT_ZONES = {
    1: "Newark Airport",
    132: "JFK Airport",
    138: "LaGuardia Airport"
}

airport_zone_ids = list(AIRPORT_ZONES.keys())

# Trips from airport
from_airport = df_clean[df_clean["PULocationID"].isin(airport_zone_ids)].copy()
from_airport["airport"] = from_airport["PULocationID"].map(AIRPORT_ZONES)
from_airport["trip_type"] = "From Airport"

# Trips to airport
to_airport = df_clean[df_clean["DOLocationID"].isin(airport_zone_ids)].copy()
to_airport["airport"] = to_airport["DOLocationID"].map(AIRPORT_ZONES)
to_airport["trip_type"] = "To Airport"

# Combine both
airport_trips = pd.concat([from_airport, to_airport], ignore_index=True)

print("Airport trip rows:", len(airport_trips))
airport_trips.head()

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


# Zone lookup table
zone_lookup = pd.read_csv("data/taxi_zone_lookup.csv")

# Combine pickup and dropoff zones, but exclude "Outside of NYC" later
all_zone_counts = (
    pd.concat([
        df_clean["PULocationID"],
        df_clean["DOLocationID"]
    ])
    .value_counts()
    .reset_index()
)

all_zone_counts.columns = ["LocationID", "total_count"]

# Add zone names
all_zone_counts = all_zone_counts.merge(
    zone_lookup[["LocationID", "Zone"]],
    on="LocationID",
    how="left"
)

# Remove Outside of NYC
all_zone_counts = all_zone_counts[all_zone_counts["Zone"] != "Outside of NYC"]

# Select top 8 most popular zones
top_zones = all_zone_counts.head(8)["LocationID"]

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

# Create radar dataframe
radar_df = pd.DataFrame({
    "LocationID": top_zones
})

radar_df["pickup"] = radar_df["LocationID"].map(pickup_counts).fillna(0)
radar_df["dropoff"] = radar_df["LocationID"].map(dropoff_counts).fillna(0)

# Add zone names
radar_df = radar_df.merge(
    zone_lookup[["LocationID", "Zone"]],
    on="LocationID",
    how="left"
)

# Sort by total activity
radar_df["total"] = radar_df["pickup"] + radar_df["dropoff"]
radar_df = radar_df.sort_values("total", ascending=False)

# Normalize values between 0 and 1
radar_df["pickup_norm"] = radar_df["pickup"] / radar_df["pickup"].max()
radar_df["dropoff_norm"] = radar_df["dropoff"] / radar_df["dropoff"].max()

# Optional: shorten long labels
radar_df["Zone_short"] = radar_df["Zone"].str.replace(" Airport", "", regex=False)

labels = radar_df["Zone_short"].tolist()
pickup_values = radar_df["pickup_norm"].tolist()
dropoff_values = radar_df["dropoff_norm"].tolist()

# Close the radar chart
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