import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from streamlit_option_menu import option_menu
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    LOWESS_AVAILABLE = True
except ImportError:
    LOWESS_AVAILABLE = False
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.dataframe_explorer import dataframe_explorer
import base64
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Rio Airbnb Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern aesthetic with glassmorphism
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Glassmorphism effect for containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Enhanced metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px 0 rgba(31, 38, 135, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px 0 rgba(31, 38, 135, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(17, 25, 40, 0.75);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    /* Headers styling */
    h1 {
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    h2, h3 {
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Success boxes */
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Charts container */
    .chart-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    
    /* Loading animation */
    .stSpinner > div {
        border-color: #667eea;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        h1 {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('listings.csv')
        
        # Data cleaning - handle price column
        if df['price'].dtype == 'object':
            df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float, errors='ignore')
        else:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        df = df[df['price'].notna() & (df['price'] > 0) & (df['price'] < 10000)]
        
        # Feature engineering
        df['price_per_review'] = df['price'] / (df['number_of_reviews'] + 1)
        df['occupancy_rate'] = (365 - df['availability_365']) / 365
        df['estimated_revenue'] = df['price'] * (365 - df['availability_365'])
        df['price_category'] = pd.qcut(df['price'], q=5, labels=['Budget', 'Economy', 'Standard', 'Premium', 'Luxury'])
        
        # Add seasonality
        if 'last_review' in df.columns:
            df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
            df['review_month'] = df['last_review'].dt.month
            df['review_season'] = df['last_review'].dt.month%12 // 3 + 1
            season_map = {1: 'Summer', 2: 'Fall', 3: 'Winter', 4: 'Spring'}
            df['review_season'] = df['review_season'].map(season_map)
        
        # Add host categories
        df['host_category'] = pd.cut(df['calculated_host_listings_count'], 
                                      bins=[0, 1, 5, 20, float('inf')],
                                      labels=['Single Property', 'Small Portfolio', 'Medium Portfolio', 'Large Portfolio'])
        
        # Calculate distance to beach (approximate using major beach coordinates)
        # Major beaches in Rio: Copacabana, Ipanema, Leblon
        beach_coords = [
            (-22.9711, -43.1823),  # Copacabana
            (-22.9838, -43.2096),  # Ipanema
            (-22.9874, -43.2232),  # Leblon
            (-23.0107, -43.3015),  # Barra da Tijuca
            (-23.0241, -43.4757),  # Recreio
        ]
        
        # Calculate minimum distance to any beach
        import math
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in km
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        
        df['distance_to_beach'] = df.apply(
            lambda row: min([haversine_distance(row['latitude'], row['longitude'], beach_lat, beach_lon) 
                           for beach_lat, beach_lon in beach_coords]),
            axis=1
        )
        
        # Add beach proximity category
        df['beach_proximity'] = pd.cut(df['distance_to_beach'], 
                                       bins=[0, 1, 3, 5, float('inf')],
                                       labels=['Beachfront (<1km)', 'Near Beach (1-3km)', 'Mid Distance (3-5km)', 'Inland (>5km)'])
        
        # Add host experience level based on number of reviews
        df['host_experience'] = pd.cut(df['number_of_reviews'], 
                                       bins=[0, 10, 50, 100, float('inf')],
                                       labels=['New (0-10 reviews)', 'Growing (11-50)', 'Experienced (51-100)', 'Veteran (>100)'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = load_data()

# Load custom CSS
load_css()

# Sidebar navigation
with st.sidebar:
    st.markdown("# 🏠 Rio Airbnb Analytics")
    st.markdown("---")
    
    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Questions Overview", "Dashboard", "Data Explorer", "Advanced Analytics", "Export Data"],
        icons=["question-circle", "speedometer2", "search", "graph-up", "download"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#667eea", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "border-radius": "10px",
                "background-color": "rgba(255, 255, 255, 0.05)",
                "transition": "all 0.3s ease"
            },
            "nav-link-selected": {"background-color": "rgba(102, 126, 234, 0.2)"},
        }
    )
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔍 Filters")
    
    # Neighborhood Group filter
    neighborhood_groups = ['All'] + list(st.session_state.df['neighbourhood_group'].unique())
    selected_neighborhood_group = st.selectbox(
        "Neighborhood Group",
        neighborhood_groups,
        key='neighborhood_group_filter'
    )
    
    # Room Type filter
    room_types = ['All'] + list(st.session_state.df['room_type'].unique())
    selected_room_type = st.selectbox(
        "Room Type",
        room_types,
        key='room_type_filter'
    )
    
    # Price Range slider
    min_price = float(st.session_state.df['price'].min())
    max_price = float(st.session_state.df['price'].quantile(0.95))
    price_range = st.slider(
        "Price Range ($)",
        min_price,
        max_price,
        (min_price, st.session_state.df['price'].quantile(0.85)),
        key='price_range_filter'
    )
    
    # Advanced filters in expander
    with st.expander("Advanced Filters"):
        min_reviews = st.slider(
            "Minimum Reviews",
            0,
            50,
            0,
            key='min_reviews_filter'
        )
        
        availability_range = st.slider(
            "Availability (days)",
            0,
            365,
            (0, 365),
            key='availability_filter'
        )
        
        host_category = st.multiselect(
            "Host Category",
            options=st.session_state.df['host_category'].unique(),
            default=st.session_state.df['host_category'].unique(),
            key='host_category_filter'
        )

# Apply filters
def filter_data(df):
    filtered_df = df.copy()
    
    if selected_neighborhood_group != 'All':
        filtered_df = filtered_df[filtered_df['neighbourhood_group'] == selected_neighborhood_group]
    
    if selected_room_type != 'All':
        filtered_df = filtered_df[filtered_df['room_type'] == selected_room_type]
    
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_range[0]) & 
        (filtered_df['price'] <= price_range[1])
    ]
    
    if 'min_reviews_filter' in st.session_state:
        filtered_df = filtered_df[filtered_df['number_of_reviews'] >= st.session_state.min_reviews_filter]
    
    if 'availability_filter' in st.session_state:
        availability_range = st.session_state.availability_filter
        filtered_df = filtered_df[
            (filtered_df['availability_365'] >= availability_range[0]) & 
            (filtered_df['availability_365'] <= availability_range[1])
        ]
    
    if 'host_category_filter' in st.session_state:
        filtered_df = filtered_df[filtered_df['host_category'].isin(st.session_state.host_category_filter)]
    
    return filtered_df

filtered_df = filter_data(st.session_state.df)

# Main content based on navigation
if selected == "Questions Overview":
    # Questions Overview Section
    st.markdown("# 📋 Analysis Questions Overview")
    st.markdown("### Track which data visualization questions are being answered in this dashboard")
    
    # Create tabs for different question categories
    q_tab1, q_tab2, q_tab3, q_tab4, q_tab5 = st.tabs([
        "🌍 Geographic", "💰 Pricing", "🧍 Host Behavior", "⭐ Reviews", "🏠 Listings"
    ])
    
    with q_tab1:
        st.markdown("### 🌍 Geographic / Spatial Questions")
        
        questions_geo = [
            {"question": "Where are the most densely concentrated Airbnb listings in Rio?", "status": "", "location": "Dashboard → Geographic Distribution"},
            {"question": "What neighborhoods have the highest average prices per night?", "status": "", "location": "Dashboard → Geographic Distribution"},
            {"question": "Are there price differences between listings near the beach vs. inland?", "status": "", "location": "Dashboard → Price Analysis"}
        ]
        
        for q in questions_geo:
            st.markdown(f"**{q['question']}**")
            st.caption(f"📍 {q['location']}")
            st.write("")  # Space
    
    with q_tab2:
        st.markdown("### 💰 Pricing & Demand Questions")
        
        questions_price = [
            {"question": "How do prices vary with room type?", "status": "", "location": "Dashboard → Price Analysis"},
            {"question": "Is there a seasonal trend in price or availability?", "status": "", "location": "Dashboard → Price Analysis & Advanced Analytics"},
            {"question": "Do hosts with more properties charge differently?", "status": "", "location": "Dashboard → Market Insights"}
        ]
        
        for q in questions_price:
            st.markdown(f"**{q['question']}**")
            st.caption(f"📍 {q['location']}")
            st.write("")  # Space
    
    with q_tab3:
        st.markdown("### 🧍 Host Behavior Questions")
        
        questions_host = [
            {"question": "What is the distribution of listings per host?", "status": "", "location": "Dashboard → Market Insights"},
            {"question": "Do experienced hosts have higher occupancy rates?", "status": "", "location": "Dashboard → Market Insights"},
            {"question": "How do professional hosts compare to casual hosts?", "status": "", "location": "Dashboard → Market Insights"}
        ]
        
        for q in questions_host:
            st.markdown(f"**{q['question']}**")
            st.caption(f"📍 {q['location']}")
            st.write("")  # Space
    
    with q_tab4:
        st.markdown("### ⭐ Ratings & Reviews Questions")
        
        questions_reviews = [
            {"question": "How do review counts vary across neighborhoods?", "status": "", "location": "Dashboard → Geographic Distribution"},
            {"question": "Is there a relationship between price and review frequency?", "status": "", "location": "Dashboard → ML Analysis"},
            {"question": "Which areas have the most active listings?", "status": "", "location": "Dashboard → Geographic Distribution"}
        ]
        
        for q in questions_reviews:
            st.markdown(f"**{q['question']}**")
            st.caption(f"📍 {q['location']}")
            st.write("")  # Space
    
    with q_tab5:
        st.markdown("### 🏠 Listing Characteristics Questions")
        
        questions_listings = [
            {"question": "What is the most common room type in Rio?", "status": "", "location": "Data Explorer → Categorical Summary"},
            {"question": "Are certain types of listings more common in specific neighborhoods?", "status": "", "location": "Dashboard → Market Insights"},
            {"question": "How do Airbnb listings cluster based on features?", "status": "", "location": "Dashboard → ML Analysis"}
        ]
        
        for q in questions_listings:
            st.markdown(f"**{q['question']}**")
            st.caption(f"📍 {q['location']}")
            st.write("")  # Space
    

elif selected == "Dashboard":
    # Header
    st.markdown("# 📊 Rio de Janeiro Airbnb Analytics Dashboard")
    st.markdown("### Real-time insights and analysis of Airbnb listings in Rio")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Listings",
            value=f"{len(filtered_df):,}",
            delta=f"{((len(filtered_df) / len(st.session_state.df)) * 100):.1f}% of total"
        )
    
    with col2:
        avg_price = filtered_df['price'].mean()
        st.metric(
            label="Average Price",
            value=f"${avg_price:.0f}",
            delta=f"{((avg_price / st.session_state.df['price'].mean()) - 1) * 100:+.1f}%"
        )
    
    with col3:
        avg_occupancy = filtered_df['occupancy_rate'].mean() * 100
        st.metric(
            label="Avg Occupancy",
            value=f"{avg_occupancy:.1f}%",
            delta=f"{(avg_occupancy - st.session_state.df['occupancy_rate'].mean() * 100):.1f}pp"
        )
    
    with col4:
        total_hosts = filtered_df['host_id'].nunique()
        st.metric(
            label="Active Hosts",
            value=f"{total_hosts:,}",
            delta=f"{((total_hosts / st.session_state.df['host_id'].nunique()) * 100):.1f}% of total"
        )
    
    style_metric_cards(background_color="rgba(255,255,255,0.05)", border_radius_px=20)
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Analysis", "🗺️ Geographic Distribution", "📊 Market Insights", "🤖 ML Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution with KDE
            fig_price = go.Figure()
            
            # Histogram
            fig_price.add_trace(go.Histogram(
                x=filtered_df['price'],
                nbinsx=50,
                name='Price Distribution',
                marker_color='rgba(59, 130, 246, 0.6)',
                showlegend=False
            ))
            
            # KDE overlay
            kde = stats.gaussian_kde(filtered_df['price'])
            x_range = np.linspace(filtered_df['price'].min(), filtered_df['price'].max(), 100)
            kde_values = kde(x_range)
            
            fig_price.add_trace(go.Scatter(
                x=x_range,
                y=kde_values * len(filtered_df) * (filtered_df['price'].max() - filtered_df['price'].min()) / 50,
                mode='lines',
                name='Density',
                line=dict(color='rgba(139, 92, 246, 1)', width=3),
                yaxis='y2'
            ))
            
            fig_price.update_layout(
                title='Price Distribution Analysis',
                xaxis_title='Price ($)',
                yaxis_title='Count',
                yaxis2=dict(overlaying='y', side='right', showgrid=False, showticklabels=False),
                height=400,
                template='plotly_dark',
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            # Price by room type
            price_by_room = filtered_df.groupby('room_type').agg({
                'price': ['mean', 'median', 'std', 'count']
            }).round(2)
            price_by_room.columns = ['Mean', 'Median', 'Std Dev', 'Count']
            price_by_room = price_by_room.sort_values('Mean', ascending=False)
            
            fig_room = go.Figure()
            
            # Box plot
            for room_type in filtered_df['room_type'].unique():
                room_data = filtered_df[filtered_df['room_type'] == room_type]['price']
                fig_room.add_trace(go.Box(
                    y=room_data,
                    name=room_type,
                    boxpoints='outliers',
                    marker_color='rgba(59, 130, 246, 0.6)',
                    line_color='rgba(59, 130, 246, 1)'
                ))
            
            fig_room.update_layout(
                title='Price Distribution by Room Type',
                yaxis_title='Price ($)',
                height=400,
                template='plotly_dark',
                showlegend=False
            )
            
            st.plotly_chart(fig_room, use_container_width=True)
        
        # Price trends over time
        if 'last_review' in filtered_df.columns and filtered_df['last_review'].notna().any():
            st.markdown("### 📅 Price Trends Over Time")
            
            # Group by month and room type
            monthly_prices = filtered_df.groupby([
                pd.Grouper(key='last_review', freq='M'),
                'room_type'
            ])['price'].mean().reset_index()
            
            fig_trends = px.line(
                monthly_prices,
                x='last_review',
                y='price',
                color='room_type',
                title='Monthly Average Price Trends by Room Type',
                labels={'price': 'Average Price ($)', 'last_review': 'Month'},
                template='plotly_dark',
                height=400
            )
            
            fig_trends.update_traces(mode='lines+markers')
            st.plotly_chart(fig_trends, use_container_width=True)
        
        # Beach proximity analysis
        st.markdown("### 🏖️ Beach Proximity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price by beach proximity
            beach_price = filtered_df.groupby('beach_proximity').agg({
                'price': ['mean', 'median', 'count']
            }).round(2)
            beach_price.columns = ['Mean Price', 'Median Price', 'Count']
            
            fig_beach_price = go.Figure()
            
            # Bar chart for average prices
            fig_beach_price.add_trace(go.Bar(
                x=beach_price.index,
                y=beach_price['Mean Price'],
                name='Average Price',
                marker_color='rgba(59, 130, 246, 0.8)',
                text=beach_price['Mean Price'].round(0),
                textposition='auto'
            ))
            
            fig_beach_price.update_layout(
                title='Average Price by Beach Proximity',
                xaxis_title='Distance Category',
                yaxis_title='Average Price ($)',
                height=400,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_beach_price, use_container_width=True)
        
        with col2:
            # Distribution of listings by beach proximity
            beach_dist = filtered_df['beach_proximity'].value_counts()
            
            fig_beach_dist = go.Figure(data=[go.Pie(
                labels=beach_dist.index,
                values=beach_dist.values,
                hole=0.4,
                marker=dict(colors=['#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe'])
            )])
            
            fig_beach_dist.update_layout(
                title='Distribution of Listings by Beach Proximity',
                height=400,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_beach_dist, use_container_width=True)
    
    with tab2:
        # Geographic visualization
        st.markdown("### 🗺️ Geographic Distribution of Listings")
        
        # Map controls
        col1, col2, col3 = st.columns(3)
        with col1:
            map_color = st.selectbox(
                "Color by",
                ['price_category', 'room_type', 'occupancy_rate']
            )
        with col2:
            map_size = st.selectbox(
                "Size by",
                ['price', 'number_of_reviews', 'estimated_revenue']
            )
        
        # Create map
        if map_color == 'occupancy_rate':
            color_scale = 'Viridis'
            hover_data = ['name', 'price', 'room_type', 'occupancy_rate']
        else:
            color_scale = None
            hover_data = ['name', 'price', 'room_type', 'number_of_reviews']
        
        fig_map = px.scatter_map(
            filtered_df,
            lat='latitude',
            lon='longitude',
            color=map_color,
            size=map_size,
            hover_name='name',
            hover_data=hover_data,
            map_style='carto-darkmatter',
            height=600,
            title=f'Listings Distribution - Colored by {map_color.replace("_", " ").title()}',
            color_continuous_scale=color_scale
        )
        
        fig_map.update_layout(
            map=dict(
                center=dict(
                    lat=filtered_df['latitude'].mean(),
                    lon=filtered_df['longitude'].mean()
                ),
                zoom=10
            ),
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Neighborhood analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Top neighborhoods by listings
            top_neighborhoods = filtered_df['neighbourhood'].value_counts().head(10)
            
            fig_neighborhoods = go.Figure([go.Bar(
                x=top_neighborhoods.values,
                y=top_neighborhoods.index,
                orientation='h',
                marker_color='rgba(139, 92, 246, 0.8)'
            )])
            
            fig_neighborhoods.update_layout(
                title='Top 10 Neighborhoods by Listing Count',
                xaxis_title='Number of Listings',
                height=400,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_neighborhoods, use_container_width=True)
        
        with col2:
            # Average price by neighborhood
            avg_price_neighborhood = filtered_df.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).head(10)
            
            fig_price_neighborhood = go.Figure([go.Bar(
                x=avg_price_neighborhood.values,
                y=avg_price_neighborhood.index,
                orientation='h',
                marker_color='rgba(59, 130, 246, 0.8)'
            )])
            
            fig_price_neighborhood.update_layout(
                title='Top 10 Most Expensive Neighborhoods',
                xaxis_title='Average Price ($)',
                height=400,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_price_neighborhood, use_container_width=True)
        
        # Review activity by neighborhood
        st.markdown("### 📊 Review Activity by Neighborhood")
        
        review_by_neighborhood = filtered_df.groupby('neighbourhood').agg({
            'number_of_reviews': ['mean', 'sum', 'count'],
            'reviews_per_month': 'mean'
        }).round(2)
        review_by_neighborhood.columns = ['Avg Reviews', 'Total Reviews', 'Listings', 'Reviews/Month']
        review_by_neighborhood = review_by_neighborhood.sort_values('Total Reviews', ascending=False).head(15)
        
        fig_reviews = go.Figure()
        
        # Bar chart for total reviews
        fig_reviews.add_trace(go.Bar(
            x=review_by_neighborhood.index,
            y=review_by_neighborhood['Total Reviews'],
            name='Total Reviews',
            marker_color='rgba(16, 185, 129, 0.8)',
            yaxis='y'
        ))
        
        # Line chart for average reviews per month
        fig_reviews.add_trace(go.Scatter(
            x=review_by_neighborhood.index,
            y=review_by_neighborhood['Reviews/Month'],
            name='Avg Reviews/Month',
            mode='lines+markers',
            marker_size=8,
            line=dict(color='rgba(239, 68, 68, 1)', width=3),
            yaxis='y2'
        ))
        
        fig_reviews.update_layout(
            title='Review Activity by Neighborhood',
            xaxis_title='Neighborhood',
            yaxis_title='Total Reviews',
            yaxis2=dict(title='Reviews per Month', overlaying='y', side='right'),
            height=500,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        fig_reviews.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_reviews, use_container_width=True)
        
        # Room type distribution by neighborhood
        st.markdown("### 🏠 Room Type Distribution by Neighborhood")
        
        # Get top neighborhoods
        top_neighborhoods_for_room = filtered_df['neighbourhood'].value_counts().head(10).index
        room_type_dist = filtered_df[filtered_df['neighbourhood'].isin(top_neighborhoods_for_room)].groupby(['neighbourhood', 'room_type']).size().unstack(fill_value=0)
        
        # Calculate percentages
        room_type_pct = room_type_dist.div(room_type_dist.sum(axis=1), axis=0) * 100
        
        fig_room_dist = go.Figure()
        
        for room_type in room_type_pct.columns:
            fig_room_dist.add_trace(go.Bar(
                name=room_type,
                x=room_type_pct.index,
                y=room_type_pct[room_type],
                text=room_type_pct[room_type].round(1).astype(str) + '%',
                textposition='inside'
            ))
        
        fig_room_dist.update_layout(
            title='Room Type Distribution in Top 10 Neighborhoods',
            xaxis_title='Neighborhood',
            yaxis_title='Percentage (%)',
            barmode='stack',
            height=500,
            template='plotly_dark',
            showlegend=True
        )
        
        fig_room_dist.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_room_dist, use_container_width=True)
        
    
    with tab3:
        # Market insights
        st.markdown("### 💡 Market Insights & Revenue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue analysis
            revenue_by_type = filtered_df.groupby('room_type').agg({
                'estimated_revenue': 'mean',
                'occupancy_rate': 'mean',
                'price': 'mean'
            }).reset_index()
            
            fig_revenue = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]]
            )
            
            # Revenue bars
            fig_revenue.add_trace(
                go.Bar(
                    x=revenue_by_type['room_type'],
                    y=revenue_by_type['estimated_revenue'],
                    name='Avg Revenue',
                    marker_color='rgba(59, 130, 246, 0.8)'
                ),
                secondary_y=False
            )
            
            # Occupancy line
            fig_revenue.add_trace(
                go.Scatter(
                    x=revenue_by_type['room_type'],
                    y=revenue_by_type['occupancy_rate'] * 100,
                    name='Occupancy %',
                    mode='lines+markers',
                    marker_size=10,
                    line=dict(color='rgba(16, 185, 129, 1)', width=3)
                ),
                secondary_y=True
            )
            
            fig_revenue.update_xaxes(title_text="Room Type")
            fig_revenue.update_yaxes(title_text="Average Revenue ($)", secondary_y=False)
            fig_revenue.update_yaxes(title_text="Occupancy Rate (%)", secondary_y=True)
            fig_revenue.update_layout(
                title='Revenue & Occupancy by Room Type',
                template='plotly_dark',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            # Host portfolio analysis
            host_analysis = filtered_df.groupby('host_category').agg({
                'price': 'mean',
                'estimated_revenue': 'mean',
                'occupancy_rate': 'mean',
                'host_id': 'count'
            }).reset_index()
            
            fig_host = go.Figure()
            
            # Bubble chart
            fig_host.add_trace(go.Scatter(
                x=host_analysis['price'],
                y=host_analysis['occupancy_rate'] * 100,
                mode='markers+text',
                marker=dict(
                    size=host_analysis['host_id'] / 10,
                    color=host_analysis['estimated_revenue'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Revenue")
                ),
                text=host_analysis['host_category'],
                textposition="top center"
            ))
            
            fig_host.update_layout(
                title='Host Portfolio Performance',
                xaxis_title='Average Price ($)',
                yaxis_title='Occupancy Rate (%)',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig_host, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔗 Feature Correlations")
        
        numerical_cols = ['price', 'number_of_reviews', 'reviews_per_month', 
                          'availability_365', 'calculated_host_listings_count',
                          'occupancy_rate', 'estimated_revenue']
        
        available_cols = [col for col in numerical_cols if col in filtered_df.columns]
        corr_matrix = filtered_df[available_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig_corr.update_layout(
            title='Feature Correlation Matrix',
            height=500,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Prepare data for analysis  
        st.markdown("### 📊 Reviews vs Price Analysis")
        
        # Prepare data with outlier removal
        analysis_df = filtered_df.copy()
        analysis_df = analysis_df[analysis_df['occupancy_rate'].notna() & analysis_df['price'].notna()]
        
        # Remove outliers using IQR method
        Q1_price = analysis_df['price'].quantile(0.25)
        Q3_price = analysis_df['price'].quantile(0.75)
        IQR_price = Q3_price - Q1_price
        price_lower = Q1_price - 1.5 * IQR_price
        price_upper = Q3_price + 1.5 * IQR_price
        
        Q1_occ = analysis_df['occupancy_rate'].quantile(0.25)
        Q3_occ = analysis_df['occupancy_rate'].quantile(0.75)
        IQR_occ = Q3_occ - Q1_occ
        occ_lower = max(0, Q1_occ - 1.5 * IQR_occ)
        occ_upper = min(1, Q3_occ + 1.5 * IQR_occ)
        
        # Apply outlier filtering
        analysis_df_clean = analysis_df[
            (analysis_df['price'] >= price_lower) & 
            (analysis_df['price'] <= price_upper) &
            (analysis_df['occupancy_rate'] >= occ_lower) & 
            (analysis_df['occupancy_rate'] <= occ_upper)
        ]
        
        # Occupancy (Low Availability) vs Price Analysis
        st.markdown("### 🏨 High Occupancy vs Price Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot: Availability vs Price
            fig_availability = go.Figure()
            
            # Sample data for better performance
            sample_size = min(2000, len(analysis_df_clean))
            sample_df = analysis_df_clean.sample(n=sample_size, random_state=42)
            
            # Add scatter plot
            fig_availability.add_trace(go.Scatter(
                x=sample_df['availability_365'],
                y=sample_df['price'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=sample_df['occupancy_rate'] * 100,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Occupancy %"),
                    opacity=0.6
                ),
                name='Listings',
                text=sample_df['name'],
                hovertemplate='<b>%{text}</b><br>' +
                              'Availability: %{x} days<br>' +
                              'Price: $%{y:.2f}<br>' +
                              'Occupancy: %{marker.color:.1f}%<br>' +
                              '<extra></extra>'
            ))
            
            # Add trend line
            if len(sample_df) > 30:
                from scipy import stats as scipy_stats
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                    sample_df['availability_365'], sample_df['price']
                )
                x_trend = np.array([0, 365])
                y_trend = slope * x_trend + intercept
                
                fig_availability.add_trace(go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    name=f'Trend (r={r_value:.3f})',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig_availability.update_layout(
                title='Availability vs Price (Low availability = High occupancy)',
                xaxis_title='Days Available (out of 365)',
                yaxis_title='Price ($)',
                height=450,
                template='plotly_dark',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="rgba(255,255,255,0.2)",
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig_availability, use_container_width=True)
            
        with col2:
            # Occupancy bands vs average price
            occupancy_bins = pd.cut(analysis_df_clean['occupancy_rate'], 
                                   bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                   labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
            
            occupancy_price = analysis_df_clean.groupby(occupancy_bins)['price'].agg(['mean', 'median', 'count']).round(2)
            
            fig_occ_bars = go.Figure()
            
            fig_occ_bars.add_trace(go.Bar(
                x=occupancy_price.index.astype(str),
                y=occupancy_price['mean'],
                name='Mean Price',
                marker_color='rgba(59, 130, 246, 0.8)',
                text=occupancy_price['mean'].round(0),
                textposition='auto'
            ))
            
            fig_occ_bars.add_trace(go.Bar(
                x=occupancy_price.index.astype(str),
                y=occupancy_price['median'],
                name='Median Price',
                marker_color='rgba(139, 92, 246, 0.8)',
                text=occupancy_price['median'].round(0),
                textposition='auto'
            ))
            
            fig_occ_bars.update_layout(
                title='Average Price by Occupancy Rate',
                xaxis_title='Occupancy Rate',
                yaxis_title='Price ($)',
                height=450,
                template='plotly_dark',
                barmode='group'
            )
            
            st.plotly_chart(fig_occ_bars, use_container_width=True)
        
        
        # Price Influence on Guest Reviews Analysis
        st.markdown("### 💬 Price Influence on Guest Reviews")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Number of reviews vs price scatter plot
            fig_reviews_price = go.Figure()
            
            # Remove outliers for cleaner visualization
            review_df = analysis_df_clean[analysis_df_clean['number_of_reviews'] > 0].copy()
            
            # Sample data
            sample_size = min(2000, len(review_df))
            sample_df = review_df.sample(n=sample_size, random_state=42)
            
            fig_reviews_price.add_trace(go.Scatter(
                x=sample_df['price'],
                y=sample_df['number_of_reviews'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=sample_df['reviews_per_month'],
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Reviews/Month"),
                    opacity=0.6
                ),
                name='Listings',
                text=sample_df['name'],
                hovertemplate='<b>%{text}</b><br>' +
                              'Price: $%{x:.2f}<br>' +
                              'Total Reviews: %{y}<br>' +
                              'Reviews/Month: %{marker.color:.2f}<br>' +
                              '<extra></extra>'
            ))
            
            # Add LOWESS trend if available
            if len(sample_df) > 30 and LOWESS_AVAILABLE:
                sorted_sample = sample_df.sort_values('price')
                lowess_result = lowess(sorted_sample['number_of_reviews'], sorted_sample['price'], frac=0.3)
                
                fig_reviews_price.add_trace(go.Scatter(
                    x=lowess_result[:, 0],
                    y=lowess_result[:, 1],
                    mode='lines',
                    name='Trend (LOWESS)',
                    line=dict(color='red', width=3)
                ))
            
            fig_reviews_price.update_layout(
                title='Price vs Number of Reviews',
                xaxis_title='Price ($)',
                yaxis_title='Number of Reviews',
                height=450,
                template='plotly_dark',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="rgba(255,255,255,0.2)",
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig_reviews_price, use_container_width=True)
            
        with col2:
            # Price categories vs review metrics
            price_categories = pd.cut(review_df['price'], 
                                    bins=[0, 50, 100, 200, 500, float('inf')],
                                    labels=['$0-50', '$51-100', '$101-200', '$201-500', '$500+'])
            
            review_metrics = review_df.groupby(price_categories).agg({
                'number_of_reviews': 'mean',
                'reviews_per_month': 'mean',
                'occupancy_rate': 'mean'
            }).round(2)
            
            fig_price_reviews = go.Figure()
            
            # Create subplots for different metrics
            fig_price_reviews.add_trace(go.Bar(
                x=review_metrics.index.astype(str),
                y=review_metrics['number_of_reviews'],
                name='Avg Reviews',
                marker_color='rgba(59, 130, 246, 0.8)',
                yaxis='y',
                offsetgroup=1
            ))
            
            fig_price_reviews.add_trace(go.Bar(
                x=review_metrics.index.astype(str),
                y=review_metrics['reviews_per_month'] * 10,  # Scale for visibility
                name='Reviews/Month (x10)',
                marker_color='rgba(139, 92, 246, 0.8)',
                yaxis='y',
                offsetgroup=2
            ))
            
            fig_price_reviews.update_layout(
                title='Review Activity by Price Range',
                xaxis_title='Price Range',
                yaxis_title='Average Count',
                height=450,
                template='plotly_dark',
                barmode='group'
            )
            
            st.plotly_chart(fig_price_reviews, use_container_width=True)
        
        # Statistical insights
        correlation_reviews = review_df['price'].corr(review_df['number_of_reviews'])
        correlation_rpm = review_df['price'].corr(review_df['reviews_per_month'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Price-Reviews Correlation", f"{correlation_reviews:.3f}")
        with col2:
            st.metric("Price-Reviews/Month Correlation", f"{correlation_rpm:.3f}")
        with col3:
            # Find price sweet spot
            if len(review_metrics) > 0:
                sweet_spot = review_metrics['number_of_reviews'].idxmax()
                st.metric("Most Reviewed Price Range", sweet_spot)
        
        
        # Host experience analysis
        st.markdown("### 👥 Host Experience vs Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Experience level performance
            exp_performance = filtered_df.groupby('host_experience').agg({
                'occupancy_rate': 'mean',
                'price': 'mean',
                'estimated_revenue': 'mean',
                'number_of_reviews': 'count'
            }).round(2)
            exp_performance.columns = ['Avg Occupancy', 'Avg Price', 'Avg Revenue', 'Listings']
            
            fig_exp_occ = go.Figure()
            
            # Bar chart for occupancy by experience
            fig_exp_occ.add_trace(go.Bar(
                x=exp_performance.index,
                y=exp_performance['Avg Occupancy'] * 100,
                name='Occupancy Rate',
                marker_color='rgba(59, 130, 246, 0.8)',
                text=(exp_performance['Avg Occupancy'] * 100).round(1),
                textposition='auto'
            ))
            
            fig_exp_occ.update_layout(
                title='Average Occupancy Rate by Host Experience',
                xaxis_title='Host Experience Level',
                yaxis_title='Occupancy Rate (%)',
                height=400,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_exp_occ, use_container_width=True)
        
        with col2:
            # Revenue by experience
            fig_exp_rev = go.Figure()
            
            # Combined bar and line chart
            fig_exp_rev.add_trace(go.Bar(
                x=exp_performance.index,
                y=exp_performance['Avg Revenue'],
                name='Avg Revenue',
                marker_color='rgba(16, 185, 129, 0.8)',
                yaxis='y'
            ))
            
            fig_exp_rev.add_trace(go.Scatter(
                x=exp_performance.index,
                y=exp_performance['Avg Price'],
                name='Avg Price',
                mode='lines+markers',
                marker_size=10,
                line=dict(color='rgba(239, 68, 68, 1)', width=3),
                yaxis='y2'
            ))
            
            fig_exp_rev.update_layout(
                title='Revenue and Price by Host Experience',
                xaxis_title='Host Experience Level',
                yaxis_title='Average Revenue ($)',
                yaxis2=dict(title='Average Price ($)', overlaying='y', side='right'),
                height=400,
                template='plotly_dark',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_exp_rev, use_container_width=True)
        
        # Minimum Nights Analysis
        st.markdown("### 🛏️ Minimum Nights & Booking Patterns Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of minimum nights
            min_nights_dist = filtered_df['minimum_nights'].value_counts().sort_index()
            
            # Group into categories for better visualization
            min_nights_categories = pd.cut(filtered_df['minimum_nights'], 
                                         bins=[0, 1, 3, 7, 30, float('inf')],
                                         labels=['1 night', '2-3 nights', '4-7 nights', '1-4 weeks', '1+ months'])
            
            min_nights_cat_dist = min_nights_categories.value_counts()
            
            fig_min_nights = go.Figure(data=[go.Pie(
                labels=min_nights_cat_dist.index,
                values=min_nights_cat_dist.values,
                hole=0.4,
                marker=dict(colors=['#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe', '#f1f5f9'])
            )])
            
            fig_min_nights.update_layout(
                title='Distribution of Minimum Nights Requirements',
                height=400,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_min_nights, use_container_width=True)
        
        with col2:
            # Price vs minimum nights analysis
            min_nights_price = filtered_df.groupby(min_nights_categories).agg({
                'price': ['mean', 'median', 'count'],
                'occupancy_rate': 'mean',
                'estimated_revenue': 'mean'
            }).round(2)
            min_nights_price.columns = ['Mean Price', 'Median Price', 'Listings', 'Avg Occupancy', 'Avg Revenue']
            
            fig_price_min_nights = go.Figure()
            
            # Bar chart for average prices by minimum nights category
            fig_price_min_nights.add_trace(go.Bar(
                x=min_nights_price.index.astype(str),
                y=min_nights_price['Mean Price'],
                name='Average Price',
                marker_color='rgba(59, 130, 246, 0.8)',
                text=min_nights_price['Mean Price'].round(0),
                textposition='auto',
                yaxis='y'
            ))
            
            # Line chart for occupancy rate
            fig_price_min_nights.add_trace(go.Scatter(
                x=min_nights_price.index.astype(str),
                y=min_nights_price['Avg Occupancy'] * 100,
                name='Occupancy Rate (%)',
                mode='lines+markers',
                marker_size=10,
                line=dict(color='rgba(16, 185, 129, 1)', width=3),
                yaxis='y2'
            ))
            
            fig_price_min_nights.update_layout(
                title='Price & Occupancy by Minimum Nights',
                xaxis_title='Minimum Nights Category',
                yaxis_title='Average Price ($)',
                yaxis2=dict(title='Occupancy Rate (%)', overlaying='y', side='right'),
                height=400,
                template='plotly_dark',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_price_min_nights, use_container_width=True)
        
        # Guest Type Segmentation
        st.markdown("#### 🎯 Guest Type Market Segmentation")
        
        # Create guest type categories
        guest_types = pd.cut(filtered_df['minimum_nights'], 
                           bins=[0, 1, 4, 15, 30, float('inf')],
                           labels=['Tourist (1 night)', 'Short Stay (2-4 nights)', 
                                 'Extended Stay (5-15 nights)', 'Monthly (16-30 nights)', 
                                 'Long-term (30+ nights)'])
        
        guest_analysis = filtered_df.copy()
        guest_analysis['guest_type'] = guest_types
        
        # Analyze each segment
        segment_analysis = guest_analysis.groupby('guest_type').agg({
            'price': 'mean',
            'estimated_revenue': 'mean',
            'occupancy_rate': 'mean',
            'number_of_reviews': 'mean',
            'id': 'count'
        }).round(2)
        segment_analysis.columns = ['Avg Price', 'Avg Revenue', 'Avg Occupancy', 'Avg Reviews', 'Listings Count']
        
        # Bubble chart for guest segments
        fig_segments = go.Figure()
        
        for segment in segment_analysis.index:
            if pd.notna(segment):  # Skip NaN values
                fig_segments.add_trace(go.Scatter(
                    x=[segment_analysis.loc[segment, 'Avg Price']],
                    y=[segment_analysis.loc[segment, 'Avg Occupancy'] * 100],
                    mode='markers+text',
                    marker=dict(
                        size=segment_analysis.loc[segment, 'Listings Count'] / 10,
                        color=segment_analysis.loc[segment, 'Avg Revenue'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Avg Revenue"),
                        opacity=0.7
                    ),
                    text=segment,
                    textposition="top center",
                    name=segment,
                    hovertemplate='<b>%{text}</b><br>' +
                                'Avg Price: $%{x:.2f}<br>' +
                                'Occupancy: %{y:.1f}%<br>' +
                                'Listings: ' + str(int(segment_analysis.loc[segment, 'Listings Count'])) + '<br>' +
                                'Avg Revenue: $' + str(int(segment_analysis.loc[segment, 'Avg Revenue'])) + '<br>' +
                                '<extra></extra>'
                ))
        
        fig_segments.update_layout(
            title='Guest Type Market Segmentation Analysis',
            xaxis_title='Average Price ($)',
            yaxis_title='Occupancy Rate (%)',
            template='plotly_dark',
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_segments, use_container_width=True)
        
        # Segment insights table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Segment Performance Metrics")
            # Display the segment analysis table with styling
            segment_display = segment_analysis.copy()
            segment_display['Avg Price'] = segment_display['Avg Price'].apply(lambda x: f"${x:.2f}")
            segment_display['Avg Revenue'] = segment_display['Avg Revenue'].apply(lambda x: f"${x:,.0f}")
            segment_display['Avg Occupancy'] = (segment_display['Avg Occupancy'] * 100).apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(
                segment_display.style.background_gradient(subset=['Listings Count'], cmap='Blues'),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 💡 Key Insights")
            
            # Calculate insights
            best_revenue_segment = segment_analysis['Avg Revenue'].idxmax()
            best_occupancy_segment = segment_analysis['Avg Occupancy'].idxmax()
            largest_segment = segment_analysis['Listings Count'].idxmax()
            
            st.info(f"🏆 **Highest Revenue**: {best_revenue_segment}")
            st.info(f"📈 **Best Occupancy**: {best_occupancy_segment}")  
            st.info(f"📊 **Largest Market**: {largest_segment}")
            
            # Market opportunity analysis
            total_listings = segment_analysis['Listings Count'].sum()
            market_share = (segment_analysis['Listings Count'] / total_listings * 100).round(1)
            
            st.markdown("**Market Share:**")
            for segment in market_share.index:
                if pd.notna(segment):
                    st.write(f"- {segment}: {market_share[segment]}%")
        
    
    with tab4:
        # ML Analysis
        st.markdown("### 🤖 Machine Learning Analysis")
        
        # Clustering analysis
        st.markdown("#### 🎯 Property Clustering")
        
        # Prepare data for clustering
        features = ['price', 'number_of_reviews', 'availability_365']
        X = filtered_df[features].fillna(0)
        
        if len(X) > 10:
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Clustering
            n_clusters = st.slider("Number of clusters", 2, 6, 4)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add to dataframe
            cluster_df = filtered_df.copy()
            cluster_df['cluster'] = clusters
            cluster_df['pca_1'] = X_pca[:, 0]
            cluster_df['pca_2'] = X_pca[:, 1]
            
            # Visualize clusters
            fig_clusters = px.scatter(
                cluster_df,
                x='pca_1',
                y='pca_2',
                color='cluster',
                hover_data=['name', 'price', 'room_type'],
                title='Property Clusters (PCA Visualization)',
                template='plotly_dark',
                height=500,
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            # Cluster characteristics
            st.markdown("#### 📊 Cluster Characteristics")
            
            cluster_summary = cluster_df.groupby('cluster').agg({
                'price': ['mean', 'count'],
                'number_of_reviews': 'mean',
                'availability_365': 'mean',
                'occupancy_rate': 'mean',
                'estimated_revenue': 'mean'
            }).round(2)
            
            cluster_summary.columns = ['Avg Price', 'Count', 'Avg Reviews', 
                                       'Avg Availability', 'Avg Occupancy', 'Avg Revenue']
            
            st.dataframe(
                cluster_summary.style.background_gradient(cmap='viridis', axis=0),
                use_container_width=True
            )
            
            # Simple cluster insights
            st.markdown("#### 💡 Key Cluster Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                best_revenue_cluster = cluster_summary['Avg Revenue'].idxmax()
                largest_cluster = cluster_summary['Count'].idxmax()
                
                st.info(f"🏆 **Highest Revenue**: Cluster {best_revenue_cluster}")
                st.info(f"📊 **Largest Segment**: Cluster {largest_cluster}")
            
            with col2:
                best_occupancy_cluster = cluster_summary['Avg Occupancy'].idxmax()
                most_expensive_cluster = cluster_summary['Avg Price'].idxmax()
                
                st.info(f"📈 **Best Occupancy**: Cluster {best_occupancy_cluster}")
                st.info(f"💰 **Most Expensive**: Cluster {most_expensive_cluster}")
            
            # Feature importance
            st.markdown("#### 📈 PCA Feature Importance & Insights")
            
            feature_importance = pd.DataFrame(
                pca.components_.T,
                columns=['PC1', 'PC2'],
                index=features
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_importance = go.Figure()
                
                for i, feature in enumerate(features):
                    fig_importance.add_trace(go.Bar(
                        x=['PC1', 'PC2'],
                        y=feature_importance.loc[feature],
                        name=feature
                    ))
                
                fig_importance.update_layout(
                    title='Feature Contributions to Principal Components',
                    xaxis_title='Principal Component',
                    yaxis_title='Contribution',
                    template='plotly_dark',
                    height=400,
                    barmode='group'
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                # PCA Analysis and Insights
                st.markdown("#### 💡 PCA Analysis & Interpretation")
                
                # Calculate explained variance
                explained_variance = pca.explained_variance_ratio_
                total_variance = explained_variance.sum()
                
                st.metric("Total Variance Explained", f"{total_variance:.1%}")
                st.metric("PC1 Variance", f"{explained_variance[0]:.1%}")
                st.metric("PC2 Variance", f"{explained_variance[1]:.1%}")
                
                st.markdown("**PCA Analysis Summary:**")
                st.write(f"• **PC1 explains {explained_variance[0]:.1%} of variance**")
                st.write(f"• **PC2 explains {explained_variance[1]:.1%} of variance**")
                st.write(f"• **Total variance captured: {total_variance:.1%}**")
            
            # Feature correlation analysis
            st.markdown("#### 🔗 Feature Correlation Analysis")
            
            correlation_matrix = filtered_df[features].corr()
            
            fig_corr_pca = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.around(correlation_matrix.values, decimals=2),
                texttemplate='%{text}',
                textfont={"size": 12}
            ))
            
            fig_corr_pca.update_layout(
                title='Feature Correlation Matrix (Used in PCA)',
                height=400,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_corr_pca, use_container_width=True)
            
            # Correlation insights
            high_correlations = []
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        direction = "positive" if corr_val > 0 else "negative"
                        high_correlations.append(f"• {features[i]} & {features[j]}: {direction} correlation ({corr_val:.2f})")
            
            if high_correlations:
                st.markdown("**Strong Feature Correlations:**")
                for corr in high_correlations:
                    st.write(corr)
            else:
                st.info("No strong correlations (>0.5) found between features - good for PCA!")

elif selected == "Data Explorer":
    st.markdown("# 🔍 Data Explorer")
    st.markdown("### Explore and filter the Airbnb listings dataset")
    
    # Interactive data explorer
    explorer_df = dataframe_explorer(filtered_df, case=False)
    
    # Display metrics for filtered data
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Filtered Listings:** {len(explorer_df):,}")
    with col2:
        st.info(f"**Average Price:** ${explorer_df['price'].mean():.2f}")
    with col3:
        st.info(f"**Total Revenue:** ${explorer_df['estimated_revenue'].sum():,.0f}")
    
    # Display the dataframe
    st.dataframe(
        explorer_df,
        use_container_width=True,
        height=600
    )
    
    # Quick stats
    st.markdown("### 📊 Quick Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Numeric Summary")
        numeric_cols = explorer_df.select_dtypes(include=[np.number]).columns
        st.dataframe(
            explorer_df[numeric_cols].describe().round(2),
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### Categorical Summary")
        cat_cols = ['room_type', 'neighbourhood_group', 'host_category', 'price_category']
        cat_summary = {}
        for col in cat_cols:
            if col in explorer_df.columns:
                cat_summary[col] = explorer_df[col].value_counts().to_dict()
        
        for col, counts in cat_summary.items():
            st.markdown(f"**{col.replace('_', ' ').title()}:**")
            for value, count in list(counts.items())[:5]:
                st.write(f"- {value}: {count} ({(count/len(explorer_df)*100):.1f}%)")

elif selected == "Advanced Analytics":
    st.markdown("# 🧠 Advanced Analytics")
    st.markdown("### Deep dive into patterns and predictions")
    
    # Time series analysis
    if 'last_review' in filtered_df.columns and filtered_df['last_review'].notna().any():
        st.markdown("### 📈 Time Series Analysis")
        
        # Prepare time series data
        ts_df = filtered_df[filtered_df['last_review'].notna()].copy()
        ts_df = ts_df.sort_values('last_review')
        
        # Monthly aggregation
        monthly_metrics = ts_df.groupby(pd.Grouper(key='last_review', freq='M')).agg({
            'price': 'mean',
            'number_of_reviews': 'count',
            'availability_365': 'mean'
        }).reset_index()
        
        # Create subplots
        fig_ts = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Average Price Trend', 'Review Activity', 'Availability Trend'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Price trend
        fig_ts.add_trace(
            go.Scatter(
                x=monthly_metrics['last_review'],
                y=monthly_metrics['price'],
                mode='lines+markers',
                name='Avg Price',
                line=dict(color='#3b82f6', width=3)
            ),
            row=1, col=1
        )
        
        # Review activity
        fig_ts.add_trace(
            go.Bar(
                x=monthly_metrics['last_review'],
                y=monthly_metrics['number_of_reviews'],
                name='Reviews',
                marker_color='#8b5cf6'
            ),
            row=2, col=1
        )
        
        # Availability
        fig_ts.add_trace(
            go.Scatter(
                x=monthly_metrics['last_review'],
                y=monthly_metrics['availability_365'],
                mode='lines+markers',
                name='Avg Availability',
                line=dict(color='#10b981', width=3)
            ),
            row=3, col=1
        )
        
        fig_ts.update_xaxes(title_text="Date", row=3, col=1)
        fig_ts.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_ts.update_yaxes(title_text="Count", row=2, col=1)
        fig_ts.update_yaxes(title_text="Days", row=3, col=1)
        
        fig_ts.update_layout(
            height=800,
            template='plotly_dark',
            showlegend=False
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # Price prediction insights
    st.markdown("### 💰 Price Optimization Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Optimal price by features
        st.markdown("#### Optimal Price Ranges")
        
        # Calculate percentiles
        percentiles = filtered_df.groupby('room_type')['price'].quantile([0.25, 0.5, 0.75]).unstack()
        
        fig_optimal = go.Figure()
        
        # Create box plot data for each room type
        room_types = percentiles.index.tolist()
        colors = ['rgba(59, 130, 246, 0.6)', 'rgba(16, 185, 129, 0.6)', 'rgba(239, 68, 68, 0.6)', 'rgba(245, 101, 101, 0.6)']
        
        for i, room_type in enumerate(room_types):
            fig_optimal.add_trace(go.Box(
                name=room_type,
                x=[room_type],  # This positions the box correctly
                q1=[percentiles.loc[room_type, 0.25]],
                median=[percentiles.loc[room_type, 0.5]],
                q3=[percentiles.loc[room_type, 0.75]],
                lowerfence=[percentiles.loc[room_type, 0.25] * 0.9],
                upperfence=[percentiles.loc[room_type, 0.75] * 1.1],
                marker_color=colors[i % len(colors)],
                boxpoints=False
            ))
        
        fig_optimal.update_layout(
            title='Optimal Price Ranges by Room Type',
            xaxis_title='Room Type',
            yaxis_title='Price ($)',
            template='plotly_dark',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_optimal, use_container_width=True)
    
    with col2:
        # Revenue maximization
        st.markdown("#### Revenue Maximization")
        
        # Calculate revenue efficiency
        revenue_df = filtered_df.copy()
        revenue_df['revenue_efficiency'] = revenue_df['estimated_revenue'] / revenue_df['price']
        
        # Top performers
        top_performers = revenue_df.nlargest(20, 'revenue_efficiency')[
            ['name', 'price', 'occupancy_rate', 'revenue_efficiency']
        ].round(2)
        
        fig_efficiency = px.scatter(
            revenue_df.sample(min(1000, len(revenue_df))),
            x='price',
            y='revenue_efficiency',
            size='estimated_revenue',
            color='room_type',
            title='Revenue Efficiency Analysis',
            labels={'revenue_efficiency': 'Revenue Efficiency (days occupied)'},
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Competitive analysis
    st.markdown("### 🏆 Competitive Analysis")
    
    # Neighborhood competition
    neighborhood_competition = filtered_df.groupby('neighbourhood').agg({
        'price': ['mean', 'std', 'count'],
        'occupancy_rate': 'mean',
        'number_of_reviews': 'mean'
    }).round(2)
    
    neighborhood_competition.columns = ['Avg Price', 'Price Std', 'Listings', 'Avg Occupancy', 'Avg Reviews']
    neighborhood_competition['Competition Score'] = (
        neighborhood_competition['Listings'] * 0.4 +
        (1 / (neighborhood_competition['Price Std'] + 1)) * 100 * 0.3 +
        neighborhood_competition['Avg Occupancy'] * 100 * 0.3
    ).round(2)
    
    top_competitive = neighborhood_competition.nlargest(10, 'Competition Score')
    
    fig_competition = go.Figure()
    
    fig_competition.add_trace(go.Bar(
        x=top_competitive.index,
        y=top_competitive['Competition Score'],
        marker_color='rgba(239, 68, 68, 0.8)',
        text=top_competitive['Competition Score'],
        textposition='auto'
    ))
    
    fig_competition.update_layout(
        title='Top 10 Most Competitive Neighborhoods',
        xaxis_title='Neighborhood',
        yaxis_title='Competition Score',
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig_competition, use_container_width=True)

elif selected == "Export Data":
    st.markdown("# 📥 Export Data")
    st.markdown("### Download filtered data in various formats")
    
    # Show current filter summary
    st.markdown("### 📋 Current Filters Applied")
    
    filter_summary = {
        "Total Listings": f"{len(filtered_df):,} / {len(st.session_state.df):,}",
        "Neighborhood Group": selected_neighborhood_group,
        "Room Type": selected_room_type,
        "Price Range": f"${price_range[0]:.0f} - ${price_range[1]:.0f}",
        "Min Reviews": st.session_state.get('min_reviews_filter', 0),
        "Availability Range": st.session_state.get('availability_filter', (0, 365))
    }
    
    col1, col2 = st.columns(2)
    for i, (key, value) in enumerate(filter_summary.items()):
        if i < 3:
            col1.info(f"**{key}:** {value}")
        else:
            col2.info(f"**{key}:** {value}")
    
    # Export options
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"airbnb_rio_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            filtered_df.to_excel(writer, sheet_name='Listings', index=False)
            
            # Add summary sheet
            summary_df = pd.DataFrame([
                ['Total Listings', len(filtered_df)],
                ['Average Price', f"${filtered_df['price'].mean():.2f}"],
                ['Total Revenue', f"${filtered_df['estimated_revenue'].sum():,.0f}"],
                ['Average Occupancy', f"{filtered_df['occupancy_rate'].mean() * 100:.1f}%"],
                ['Date Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ], columns=['Metric', 'Value'])
            
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        excel_data = output.getvalue()
        
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"airbnb_rio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        # JSON export
        json_str = filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            label="🔧 Download JSON",
            data=json_str,
            file_name=f"airbnb_rio_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Preview exported data
    st.markdown("### 👁️ Preview Exported Data")
    
    preview_rows = st.slider("Number of rows to preview", 5, 50, 10)
    st.dataframe(filtered_df.head(preview_rows), use_container_width=True)
    
    # Export statistics
    st.markdown("### 📊 Export Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Data Summary")
        st.write(f"- **Rows:** {len(filtered_df):,}")
        st.write(f"- **Columns:** {len(filtered_df.columns)}")
        st.write(f"- **Memory Usage:** {filtered_df.memory_usage().sum() / 1024**2:.2f} MB")
        st.write(f"- **Date Range:** {filtered_df['last_review'].min()} to {filtered_df['last_review'].max()}")
    
    with col2:
        st.markdown("#### Column Information")
        column_info = pd.DataFrame({
            'Column': filtered_df.columns,
            'Type': filtered_df.dtypes.astype(str),
            'Non-Null Count': filtered_df.count(),
            'Null %': ((filtered_df.isnull().sum() / len(filtered_df)) * 100).round(2)
        })
        
        st.dataframe(
            column_info.style.background_gradient(subset=['Null %'], cmap='Reds'),
            use_container_width=True,
            height=300
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #64748b;'>
        <p>Camila Ilges, Eric Monteiro, Lucca Tisser, Thiago Thomas</p>
    </div>
    """,
    unsafe_allow_html=True
)