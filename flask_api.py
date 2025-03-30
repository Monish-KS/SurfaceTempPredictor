# flask_api.py
from flask import Flask, request, jsonify
import pandas as pd
import os
from pathlib import Path
import json

app = Flask(__name__)


# Check if markdown files exist and create them if they don't
def ensure_markdown_files():
    markdown_dir = Path("WorldTemperatureViewer")
    if not markdown_dir.exists():
        markdown_dir.mkdir(parents=True, exist_ok=True)

    about_md = markdown_dir / "About.md"
    if not about_md.exists():
        about_md.write_text(
            """
# About World Temperature Viewer

The World Temperature Viewer (WTV) is an application that allows users to explore and analyze global temperature data. It provides visualizations and forecasts of temperature trends around the world.

This application was developed as part of the NASA Mission: Pale Blue Dot Visualization Challenge.
        """
        )

    challenge_md = markdown_dir / "Challenge.md"
    if not challenge_md.exists():
        challenge_md.write_text(
            """
# Problem Statement

Climate change is one of the most pressing challenges facing our planet today. Understanding temperature patterns and trends is crucial for developing effective strategies to mitigate the impacts of global warming.

The World Temperature Viewer aims to provide:

1. Historical temperature data analysis
2. Temperature forecasting for various regions
3. Visual representations of temperature patterns
4. Insights into climate trends and anomalies
        """
        )


# Call this function when the Flask app starts
ensure_markdown_files()


# Shared data loading function (similar to your Streamlit code)
def load_data_api():
    # World Data
    world_path = os.path.join(os.getcwd(), "Data", "WorldData.csv")

    # Check if the data file exists
    if not os.path.exists(world_path):
        raise FileNotFoundError(f"Data file not found: {world_path}")

    world_data = pd.read_csv(world_path)
    world_data["dt"] = pd.to_datetime(world_data["dt"])
    world_data["Year"] = world_data["dt"].dt.year
    return world_data


# API endpoints
@app.route("/api/temperature/stats", methods=["GET"])
def get_temperature_stats():
    """Return temperature statistics for a given country or city"""
    country = request.args.get("country", default=None, type=str)
    city = request.args.get("city", default=None, type=str)

    try:
        world_data = load_data_api()

        filtered_data = world_data
        if country:
            filtered_data = filtered_data[filtered_data["Country"] == country]
        if city:
            filtered_data = filtered_data[filtered_data["City"] == city]

        if filtered_data.empty:
            return jsonify({"error": "No data found for the specified parameters"}), 404

        # Calculate basic statistics
        stats = {
            "average_temperature": filtered_data["AverageTemperature"].mean(),
            "max_temperature": filtered_data["AverageTemperature"].max(),
            "min_temperature": filtered_data["AverageTemperature"].min(),
            "records_count": len(filtered_data),
            "date_range": {
                "start": filtered_data["dt"].min().strftime("%Y-%m-%d"),
                "end": filtered_data["dt"].max().strftime("%Y-%m-%d"),
            },
        }

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/countries", methods=["GET"])
def get_countries():
    """Return list of all countries in the dataset"""
    try:
        world_data = load_data_api()
        countries = world_data["Country"].unique().tolist()
        return jsonify({"countries": countries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cities", methods=["GET"])
def get_cities():
    """Return list of cities, optionally filtered by country"""
    country = request.args.get("country", default=None, type=str)

    try:
        world_data = load_data_api()

        if country:
            cities = (
                world_data[world_data["Country"] == country]["City"].unique().tolist()
            )
        else:
            cities = world_data["City"].unique().tolist()

        return jsonify({"cities": cities})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app when this script is executed directly
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
