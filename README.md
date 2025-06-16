# Rio de Janeiro Airbnb Analytics Dashboard

https://rio-airbnb.streamlit.app/

A comprehensive Streamlit-based analytics dashboard for visualizing and analyzing Airbnb listings data in Rio de Janeiro. This interactive web application provides deep insights into pricing trends, geographic distribution, market dynamics, and revenue optimization opportunities.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)

## Features

### 📊 Dashboard Overview
- **Key Metrics**: Total listings, average prices, occupancy rates, and active hosts
- **Price Analysis**: Distribution analysis with KDE overlay and room type comparisons
- **Geographic Visualization**: Interactive maps with customizable color and size encoding
- **Market Insights**: Revenue analysis, host portfolio performance, and correlation heatmaps
- **ML Analysis**: K-means clustering with PCA visualization and feature importance

### 🔍 Data Explorer
- Interactive data filtering and exploration
- Real-time statistics and summaries
- Categorical and numerical data insights
- Advanced filtering capabilities

### 🧠 Advanced Analytics
- Time series analysis of pricing and review trends
- Revenue optimization insights
- Competitive analysis by neighborhood
- Price efficiency scoring

### 📥 Data Export
- Multiple export formats (CSV, Excel, JSON)
- Filtered data download with current filter summary
- Export statistics and metadata

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd airbnb-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the dashboard**
   Open your browser to `http://localhost:8501`

## Requirements

- Python 3.7+
- Streamlit 1.28.0+
- Pandas 2.0.0+
- Plotly 5.15.0+
- Scikit-learn 1.3.0+
- NumPy 1.24.0+
- SciPy 1.14.0+

See `requirements.txt` for complete dependency list.

## Data

The dashboard analyzes Rio de Janeiro Airbnb listings data with the following key features:
- Property details (name, location, room type)
- Pricing information
- Host information and portfolio size
- Review metrics and availability
- Calculated features (occupancy rate, estimated revenue, price categories)

## Usage

### Navigation
Use the sidebar menu to navigate between different sections:
- **Dashboard**: Main analytics and visualizations
- **Data Explorer**: Interactive data exploration
- **Advanced Analytics**: Deep-dive analysis and predictions
- **Export Data**: Download filtered datasets

### Filtering
Apply filters in the sidebar to focus your analysis:
- Neighborhood groups and specific neighborhoods
- Room types (Entire home/apt, Private room, Shared room)
- Price ranges with interactive sliders
- Minimum review thresholds
- Availability periods
- Host categories

### Visualizations
All charts are interactive with:
- Hover tooltips for detailed information
- Zoom and pan capabilities
- Download options for static images
- Responsive design for different screen sizes

## Technical Architecture

### Data Processing
- Automated data cleaning and validation
- Feature engineering for derived metrics
- Real-time filtering with caching optimization
- Error handling for robust data loading

### Visualization Engine
- Plotly-based interactive charts
- Custom dark theme styling
- Glassmorphism design effects
- Responsive layout with CSS Grid

### Machine Learning
- K-means clustering for property segmentation
- Principal Component Analysis (PCA) for dimensionality reduction
- Revenue optimization analysis
- Competitive scoring algorithms

## Contributing

This project was developed by:
- **Camila Ilges**
- **Eric Monteiro** 
- **Lucca Tisser**
- **Thiago Thomas**

## License

This project is for educational purposes as part of a data visualization course.
