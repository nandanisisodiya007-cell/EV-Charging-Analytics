import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# TASK 6: OPTIMAL LOCATION PLANNING
# ==========================================

INPUT_FILE = "Task 6.xlsx"
OUTPUT_DIR = Path("task6_output")

DEMAND_WEIGHT = 0.70
ENERGY_WEIGHT = 0.30

OUTPUT_DIR.mkdir(exist_ok=True)


def normalize(series):
    """Convert values to a 0-100 score."""
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(50.0, index=series.index)

    return (series - minimum) / (maximum - minimum) * 100


def find_sheet(excel, possible_names):
    """Find a worksheet without depending on capitalization."""
    sheet_lookup = {
        name.strip().lower(): name
        for name in excel.sheet_names
    }

    for name in possible_names:
        if name.lower() in sheet_lookup:
            return sheet_lookup[name.lower()]

    raise ValueError(
        f"Could not find worksheet: {possible_names}. "
        f"Available sheets: {excel.sheet_names}"
    )


def find_column(data, possible_names):
    """Find a column using possible column-name variations."""
    lookup = {
        str(column).strip().lower(): column
        for column in data.columns
    }

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def main():
    print("Starting Task 6 analysis...")

    # Load the existing Excel workbook
    excel = pd.ExcelFile(INPUT_FILE)

    print("\nWorksheets found:")
    print(excel.sheet_names)

    sessions_sheet = find_sheet(
        excel, ["sessions"]
    )
    stations_sheet = find_sheet(
        excel, ["Station", "stations"]
    )
    chargers_sheet = find_sheet(
        excel, ["Charger", "chargers"]
    )

    sessions = pd.read_excel(
        excel, sheet_name=sessions_sheet
    )
    stations = pd.read_excel(
        excel, sheet_name=stations_sheet
    )
    chargers = pd.read_excel(
        excel, sheet_name=chargers_sheet
    )

    # Identify important columns
    session_region = find_column(
        sessions, ["Region", "Location", "City", "Area"]
    )
    station_region = find_column(
        stations, ["Region", "Location", "City", "Area"]
    )

    customer_col = find_column(
        sessions, ["Customer ID", "Customer_ID", "CustomerID"]
    )
    energy_col = find_column(
        sessions, ["Energy", "Energy (kWh)", "Energy_kWh", "kWh"]
    )
    session_station_col = find_column(
        sessions, ["Station ID", "Station_ID", "StationID"]
    )
    station_id_col = find_column(
        stations, ["Station ID", "Station_ID", "StationID"]
    )

    charger_station_col = find_column(
        chargers, ["Station ID", "Station_ID", "StationID"]
    )
    charger_power_col = find_column(
        chargers, ["max_power_kw","Power", "Power (kW)", "Power_kW", "Charger Power"]
    )

    # If sessions do not contain a region, map through Station ID
    if session_region is None:
        if (
            session_station_col is None
            or station_id_col is None
            or station_region is None
        ):
            raise ValueError(
                "Cannot connect sessions to regions. "
                "Check the region and Station ID column names."
            )

        station_lookup = stations[
            [station_id_col, station_region]
        ].drop_duplicates()

        sessions = sessions.merge(
            station_lookup,
            left_on=session_station_col,
            right_on=station_id_col,
            how="left",
            suffixes=("", "_station")
        )

        session_region = station_region

    if station_region is None:
        raise ValueError(
            "A region or location column was not found in Station."
        )

    # Report column mappings for checking
    print("\nDetected columns:")
    print("Session region:", session_region)
    print("Station region:", station_region)
    print("Customer ID:", customer_col)
    print("Energy:", energy_col)
    print("Station ID:", station_id_col)
    print("Charger power:", charger_power_col)
    print("All Station columns:", stations.columns.tolist())
    print("All Charger columns:", chargers.columns.tolist())

    # Prepare numeric energy data
    if energy_col is None:
        raise ValueError(
            "Energy column not found. Update the energy column "
            "names in the code to match your workbook."
        )

    sessions["_energy_value"] = pd.to_numeric(
        sessions[energy_col], errors="coerce"
    ).fillna(0)

    # ==========================================
    # 1. REGIONAL ANALYSIS
    # ==========================================

    aggregations = {
        "Charging Sessions": (session_region, "size"),
        "Energy Delivered (kWh)": (
            "_energy_value", "sum"
        )
    }

    if customer_col is not None:
        aggregations["Unique Customers"] = (
            customer_col, "nunique"
        )
    else:
        print(
            "WARNING: Customer ID was not found. "
            "Using session count as a fallback."
        )
        aggregations["Unique Customers"] = (
            session_region, "size"
        )

    regional = sessions.groupby(
        session_region
    ).agg(**aggregations).reset_index()

    regional.rename(
        columns={session_region: "Region"},
        inplace=True
    )

    # Count existing stations in each region
    if station_id_col is not None:
        station_counts = stations.groupby(
            station_region
        )[station_id_col].nunique()
    else:
        station_counts = stations.groupby(
            station_region
        ).size()

    station_counts = station_counts.rename(
        "Existing Stations"
    )

    regional = regional.merge(
        station_counts,
        left_on="Region",
        right_index=True,
        how="left"
    )

    regional["Existing Stations"] = (
        regional["Existing Stations"].fillna(0)
    )

    # Calculate regional scores
    regional["Demand Score"] = (
        normalize(regional["Charging Sessions"]) * 0.50
        + normalize(regional["Unique Customers"]) * 0.50
    )

    regional["Energy Score"] = normalize(
        regional["Energy Delivered (kWh)"]
    )

    regional["Regional Final Score"] = (
        regional["Demand Score"] * DEMAND_WEIGHT
        + regional["Energy Score"] * ENERGY_WEIGHT
    )

    regional = regional.sort_values(
        "Regional Final Score",
        ascending=False
    ).reset_index(drop=True)

    regional["Regional Rank"] = (
        regional["Regional Final Score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    regional.to_csv(
        OUTPUT_DIR / "Regional_Analysis.csv",
        index=False
    )

    # ==========================================
    # 2. CHARGER SUPPLY ANALYSIS
    # ==========================================

    if (
        charger_station_col is not None
        and charger_power_col is not None
        and station_id_col is not None
    ):
        chargers[charger_power_col] = pd.to_numeric(
            chargers[charger_power_col],
            errors="coerce"
        ).fillna(0)

        power_by_station = chargers.groupby(
            charger_station_col
        )[charger_power_col].sum()

        station_power = stations.copy()
        station_power["_power"] = (
            station_power[station_id_col]
            .map(power_by_station)
            .fillna(0)
        )

        regional_power = station_power.groupby(
            station_region
        )["_power"].sum().rename(
            "Existing Charger Power (kW)"
        )

        regional = regional.merge(
            regional_power,
            left_on="Region",
            right_index=True,
            how="left"
        )

        regional["Existing Charger Power (kW)"] = (
            regional["Existing Charger Power (kW)"].fillna(0)
        )
    else:
        print(
            "WARNING: Could not map charger power to stations. "
            "Review the charger and station ID column names."
        )

    regional.to_csv(
        OUTPUT_DIR / "Regional_Analysis.csv",
        index=False
    )

    # ==========================================
    # 3. LOCATION EXPANSION PRIORITIES
    # ==========================================

    # This ranks regions as candidate areas unless your workbook
    # has a separate candidate-location field.
    location_col = find_column(
        stations,
        ["Candidate Location", "Location", "Site", "Region", "City", "Area"]
    )

    if location_col is None:
        raise ValueError(
            "No location column found in the Station worksheet."
        )

    # Aggregate demand at the matching location level
    if location_col in sessions.columns:
        demand_location_col = location_col
    else:
        demand_location_col = session_region

    location_aggregations = {
        "Charging Sessions": (
            demand_location_col, "size"
        ),
        "Energy Delivered (kWh)": (
            "_energy_value", "sum"
        )
    }

    if customer_col is not None:
        location_aggregations["Unique Customers"] = (
            customer_col, "nunique"
        )
    else:
        location_aggregations["Unique Customers"] = (
            demand_location_col, "size"
        )

    location_demand = sessions.groupby(
        demand_location_col
    ).agg(**location_aggregations).reset_index()

    location_demand.rename(
        columns={demand_location_col: "Location"},
        inplace=True
    )

    # Count existing stations per location
    if station_id_col is not None:
        location_supply = stations.groupby(
            location_col
        )[station_id_col].nunique().reset_index(
            name="Existing Stations"
        )
    else:
        location_supply = stations.groupby(
            location_col
        ).size().reset_index(name="Existing Stations")

    location_supply.rename(
        columns={location_col: "Location"},
        inplace=True
    )

    location = location_demand.merge(
        location_supply,
        on="Location",
        how="outer"
    ).fillna(0)

    location["Demand Score"] = (
        normalize(location["Charging Sessions"]) * 0.50
        + normalize(location["Unique Customers"]) * 0.50
    )

    location["Energy Score"] = normalize(
        location["Energy Delivered (kWh)"]
    )

    location["Supply Gap Score"] = (
        100 - normalize(location["Existing Stations"])
    )

    # Location weights are an example formula.
    # Replace these weights if your Excel formula differs.
    location["Final Location Score"] = (
        location["Demand Score"] * 0.60
        + location["Energy Score"] * 0.25
        + location["Supply Gap Score"] * 0.15
    )

    location = location.sort_values(
        "Final Location Score",
        ascending=False
    ).reset_index(drop=True)

    location["Expansion Priority"] = (
        location["Final Location Score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    location.to_csv(
        OUTPUT_DIR / "Expansion_Priorities.csv",
        index=False
    )

    # ==========================================
    # 4. CHARTS
    # ==========================================

    plt.figure(figsize=(11, 6))
    plt.bar(
        regional["Region"].astype(str),
        regional["Regional Final Score"]
    )
    plt.title("Regional Final Scores")
    plt.xlabel("Region")
    plt.ylabel("Score (0-100)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "regional_final_scores.png",
        dpi=150
    )
    plt.show()
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.barh(
        location["Location"].astype(str),
        location["Final Location Score"]
    )
    plt.title("Location Expansion Priorities")
    plt.xlabel("Final Location Score")
    plt.ylabel("Location")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "location_expansion_priorities.png",
        dpi=150
    )
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(
        regional["Existing Stations"],
        regional["Charging Sessions"]
    )
    for _, row in regional.iterrows():
        plt.annotate(
            str(row["Region"]),
            (
                row["Existing Stations"],
                row["Charging Sessions"]
            ),
            fontsize=8
        )
    plt.title("Charging Demand vs Existing Stations")
    plt.xlabel("Existing Stations")
    plt.ylabel("Charging Sessions")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "demand_vs_supply.png",
        dpi=150
    )
    plt.show()
    plt.close()

    # ==========================================
    # 5. EXECUTIVE SUMMARY
    # ==========================================

    summary = [
        "TASK 6: OPTIMAL LOCATION PLANNING FOR NEW STATIONS",
        "",
        "Objective:",
        "Analyze charging demand and existing station supply to identify",
        "regions and locations that may be priorities for expansion.",
        "",
        "Top 3 regions:",
    ]

    for _, row in regional.head(3).iterrows():
        summary.append(
            f"- {row['Region']}: regional score "
            f"{row['Regional Final Score']:.2f}"
        )

    summary += ["", "Top 5 expansion priorities:"]

    for _, row in location.head(5).iterrows():
        summary.append(
            f"- {row['Location']}: final location score "
            f"{row['Final Location Score']:.2f}"
        )

    summary += [
        "",
        "Method:",
        "Regional score = 70% demand score + 30% energy score.",
        "Location score = 60% demand + 25% energy + 15% supply gap.",
        "",
        "Important:",
        "Verify detected columns and scoring weights against the Excel workbook.",
        "These rankings are analytical recommendations, not proof of actual",
        "site suitability. Land cost, grid capacity, traffic, and permitting",
        "should also be considered before choosing a real station site."
    ]

    with open(
        OUTPUT_DIR / "Executive_Summary.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(summary))

    # ==========================================
    # 6. DISPLAY RESULTS
    # ==========================================

    print("\nTOP REGIONS")
    print(
        regional[
            ["Region", "Regional Final Score", "Unique Customers"]
        ].head(3).to_string(index=False)
    )

    print("\nTOP LOCATION PRIORITIES")
    print(
        location[
            ["Location", "Final Location Score", "Expansion Priority"]
        ].head(10).to_string(index=False)
    )

    print("\nAnalysis completed.")
    print("Output files saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()

