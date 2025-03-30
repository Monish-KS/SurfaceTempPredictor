# 🪐 Surface Temperature Predictor

## Introduction
The project utilizes the **Global Land Temperature By City** dataset from Berkeley Earth, available on Kaggle, and employs the **Meta AI Prophet Framework** for temperature forecasting.

## Key Features
- **Dynamic Visualizations**: Generate pie charts, geographic maps, and statistical summaries to explore temperature data interactively.
- **Global and Regional Insights**: Analyze temperature trends for cities worldwide and detailed U.S. state-level data.
- **Forecasting Capabilities**: Predict future temperature trends using machine learning models.
- **Streamlit-Powered Interface**: A simple and accessible web-based application for users of all technical backgrounds.
- **Docker Support**: Easily deploy the application using Docker for a consistent runtime environment.

## Installation and Setup
Follow these steps to set up and run the World Temperature Viewer locally or in a containerized environment.

### Prerequisites
- Python 3.8 or higher
- Docker (optional, for containerized deployment)
- Recommended: A virtual environment (e.g., `venv` or `conda`)

### Local Setup

#### Clone the Repository
```bash
git clone https://github.com/your-repo/WorldTemperatureViewer.git
cd WorldTemperatureViewer
```

#### Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

#### Run the Application
Start the Streamlit application:
```bash
streamlit run Vizualisations.py
```

#### Access the Application
Once the application is running, open the provided local URL (e.g., `http://localhost:8501`) in your browser.

### Docker Setup

#### Build the Docker Image
Navigate to the project directory and build the Docker image:
```bash
docker build -t world-temperature-viewer .
```

#### Run the Docker Container
Start the application in a Docker container:
```bash
docker run -p 8501:8501 world-temperature-viewer
```

#### Access the Application
Open your browser and navigate to `http://localhost:8501` to use the application.

## Folder Structure
The repository is organized as follows:
```
WorldTemperatureViewer/
├── data/                     # Contains sample datasets or links to external datasets
├── notebooks/                # Jupyter notebooks for exploratory data analysis
├── models/                   # Trained forecasting models and configurations
├── src/                      # Source code for data processing and visualization
├── Vizualisations.py         # Main Streamlit application file
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration for containerized deployment
├── README.md                 # Project documentation
```

## Dataset
The application uses the **Global Land Temperature By City** dataset, which can be downloaded from Kaggle:
[Berkeley Earth Dataset](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)

## Methodology
The dataset was divided into training (1970–2013) and testing (2010–2013) sets. The **Prophet Framework** was used for forecasting with the following parameters:
```python
{
    'growth': 'linear',
    'seasonality_mode': 'additive',
    'seasonality_prior_scale': 10.0,
    'holidays_prior_scale': 10.0,
    'changepoint_prior_scale': 0.05,
    'mcmc_samples': 0,
    'interval_width': 0.8,
    'uncertainty_samples': 1000,
    'stan_backend': None
}
```
The model achieved an **Average Mean Absolute Error (1.3)** and **Average Mean Squared Error (7.1)**, demonstrating robust performance.


## Feedback and Contributions
We welcome your feedback and contributions to improve the project. 

Thank you for supporting this initiative to promote climate awareness and sustainable practices!