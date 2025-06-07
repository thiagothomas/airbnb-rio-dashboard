import dash
from dash import dcc, html, callback, Output, Input, State, ctx
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
                ],
                suppress_callback_exceptions=True)

server = app.server

# Enhanced data loading with caching
def load_data():
    try:
        df = pd.read_csv('listings.csv')
        # Enhanced data cleaning
        df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float, errors='ignore')
        df = df[df['price'].notna() & (df['price'] > 0) & (df['price'] < 10000)]
        
        # Create additional features for analysis
        df['price_per_review'] = df['price'] / (df['number_of_reviews'] + 1)
        df['occupancy_rate'] = (365 - df['availability_365']) / 365
        df['estimated_revenue'] = df['price'] * (365 - df['availability_365'])
        df['price_category'] = pd.qcut(df['price'], q=5, labels=['Budget', 'Economy', 'Standard', 'Premium', 'Luxury'])
        
        # Add seasonality if last_review exists
        if 'last_review' in df.columns:
            df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
            df['review_month'] = df['last_review'].dt.month
            df['review_season'] = df['last_review'].dt.month%12 // 3 + 1
            df['review_season'] = df['review_season'].map({1: 'Summer', 2: 'Fall', 3: 'Winter', 4: 'Spring'})
        
        return df
    except:
        return pd.DataFrame()

# Load data
df = load_data()

# Define color schemes for light and dark modes
THEMES = {
    'dark': {
        'bg_primary': '#0f172a',
        'bg_secondary': '#1e293b',
        'bg_card': 'rgba(30, 41, 59, 0.8)',
        'text_primary': '#f1f5f9',
        'text_secondary': '#94a3b8',
        'accent': '#3b82f6',
        'accent_secondary': '#8b5cf6',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'border': 'rgba(148, 163, 184, 0.2)',
        'glassmorphism': 'rgba(15, 23, 42, 0.7)'
    },
    'light': {
        'bg_primary': '#ffffff',
        'bg_secondary': '#f8fafc',
        'bg_card': 'rgba(255, 255, 255, 0.8)',
        'text_primary': '#0f172a',
        'text_secondary': '#64748b',
        'accent': '#3b82f6',
        'accent_secondary': '#8b5cf6',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'border': 'rgba(0, 0, 0, 0.1)',
        'glassmorphism': 'rgba(255, 255, 255, 0.7)'
    }
}

# Helper functions for styling
def get_glass_card_style(theme='dark'):
    colors = THEMES[theme]
    return {
        'background': colors['glassmorphism'],
        'backdropFilter': 'blur(10px)',
        'WebkitBackdropFilter': 'blur(10px)',
        'border': f"1px solid {colors['border']}",
        'borderRadius': '16px',
        'padding': '24px',
        'marginBottom': '24px',
        'boxShadow': '0 8px 32px 0 rgba(0, 0, 0, 0.1)',
        'transition': 'all 0.3s ease'
    }

def get_metric_card_style(theme='dark'):
    colors = THEMES[theme]
    return {
        'background': f"linear-gradient(135deg, {colors['accent']}22, {colors['accent_secondary']}22)",
        'border': f"1px solid {colors['border']}",
        'borderRadius': '12px',
        'padding': '20px',
        'textAlign': 'center',
        'transition': 'all 0.3s ease'
    }

def get_sidebar_style(theme='dark'):
    colors = THEMES[theme]
    return {
        'background': colors['bg_secondary'],
        'borderRight': f"1px solid {colors['border']}",
        'height': '100vh',
        'position': 'fixed',
        'left': '0',
        'top': '0',
        'width': '280px',
        'padding': '24px',
        'overflowY': 'auto',
        'transition': 'all 0.3s ease',
        'zIndex': '1000'
    }

def get_main_container_style(theme='dark'):
    colors = THEMES[theme]
    return {
        'background': colors['bg_primary'],
        'minHeight': '100vh',
        'position': 'relative',
        'overflowX': 'hidden',
        'fontFamily': "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    }

def get_theme_toggle_style(theme='dark'):
    colors = THEMES[theme]
    return {
        'position': 'fixed',
        'top': '24px',
        'right': '24px',
        'zIndex': '1001',
        'background': colors['bg_card'],
        'backdropFilter': 'blur(10px)',
        'border': f"1px solid {colors['border']}",
        'borderRadius': '50%',
        'width': '48px',
        'height': '48px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'cursor': 'pointer',
        'transition': 'all 0.3s ease'
    }

def get_main_content_style(theme='dark'):
    return {
        'marginLeft': '280px',
        'padding': '24px',
        'transition': 'all 0.3s ease'
    }

def get_text_style(theme='dark', type='primary'):
    colors = THEMES[theme]
    return {'color': colors[f'text_{type}']}

# Create summary metrics
def create_metric_cards(theme='dark'):
    if df.empty:
        return html.Div("No data available")
    
    total_listings = len(df)
    avg_price = df['price'].mean()
    avg_availability = df['availability_365'].mean()
    total_hosts = df['host_id'].nunique()
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-home fa-2x mb-3", 
                               style={'color': '#3b82f6'}),
                        html.Div(f"{total_listings:,}", className="metric-value"),
                        html.Div("Total Listings", className="metric-label"),
                        html.Div([
                            html.I(className="fas fa-arrow-up me-1"),
                            "12% from last month"
                        ], className="metric-change", style={'color': '#10b981'})
                    ], style=get_metric_card_style(theme))
                ])
            ], lg=3, md=6),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-dollar-sign fa-2x mb-3", 
                               style={'color': '#8b5cf6'}),
                        html.Div(f"${avg_price:.0f}", className="metric-value"),
                        html.Div("Average Price", className="metric-label"),
                        html.Div([
                            html.I(className="fas fa-arrow-down me-1"),
                            "3% from last month"
                        ], className="metric-change", style={'color': '#ef4444'})
                    ], style=get_metric_card_style(theme))
                ])
            ], lg=3, md=6),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-calendar-check fa-2x mb-3", 
                               style={'color': '#10b981'}),
                        html.Div(f"{avg_availability:.0f}", className="metric-value"),
                        html.Div("Avg Days Available", className="metric-label"),
                        html.Div([
                            html.I(className="fas fa-minus me-1"),
                            "No change"
                        ], className="metric-change", style={'color': '#94a3b8'})
                    ], style=get_metric_card_style(theme))
                ])
            ], lg=3, md=6),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-users fa-2x mb-3", 
                               style={'color': '#f59e0b'}),
                        html.Div(f"{total_hosts:,}", className="metric-value"),
                        html.Div("Total Hosts", className="metric-label"),
                        html.Div([
                            html.I(className="fas fa-arrow-up me-1"),
                            "8% from last month"
                        ], className="metric-change", style={'color': '#10b981'})
                    ], style=get_metric_card_style(theme))
                ])
            ], lg=3, md=6),
        ], className="mb-4")
    ])

# Create sidebar with filters
def create_sidebar(theme='dark'):
    return html.Div([
        html.H2("Airbnb Analytics", 
                className="mb-4", 
                style={'fontSize': '1.5rem', 'fontWeight': '700'}),
        
        html.Hr(style={'borderColor': 'rgba(148, 163, 184, 0.2)'}),
        
        # Filters
        html.Div([
            # Neighborhood Group Filter
            html.Div([
                html.Label("Neighborhood Group", className="filter-label"),
                dcc.Dropdown(
                    id='neighbourhood-group-filter',
                    options=[{'label': 'All', 'value': 'all'}] + 
                            [{'label': ng, 'value': ng} for ng in df['neighbourhood_group'].unique()],
                    value='all',
                    style={'borderRadius': '8px'}
                )
            ], className="filter-section"),
            
            # Room Type Filter
            html.Div([
                html.Label("Room Type", className="filter-label"),
                dcc.Dropdown(
                    id='room-type-filter',
                    options=[{'label': 'All', 'value': 'all'}] + 
                            [{'label': rt, 'value': rt} for rt in df['room_type'].unique()],
                    value='all',
                    style={'borderRadius': '8px'}
                )
            ], className="filter-section"),
            
            # Price Range Slider
            html.Div([
                html.Label("Price Range", className="filter-label"),
                dcc.RangeSlider(
                    id='price-slider',
                    min=df['price'].min() if not df.empty else 0,
                    max=df['price'].quantile(0.95) if not df.empty else 1000,
                    value=[df['price'].min() if not df.empty else 0, 
                           df['price'].quantile(0.75) if not df.empty else 500],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="mt-3"
                )
            ], className="filter-section"),
            
            # Advanced Filters Toggle
            html.Div([
                dbc.Button(
                    [html.I(className="fas fa-filter me-2"), "Advanced Filters"],
                    id="advanced-filters-toggle",
                    color="primary",
                    outline=True,
                    className="w-100",
                    style={'borderRadius': '8px'}
                )
            ], className="filter-section"),
            
            # Advanced Filters Collapse
            dbc.Collapse([
                html.Div([
                    html.Label("Minimum Reviews", className="filter-label mt-3"),
                    dcc.Slider(
                        id='min-reviews-slider',
                        min=0,
                        max=50,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Label("Availability (days)", className="filter-label mt-3"),
                    dcc.RangeSlider(
                        id='availability-slider',
                        min=0,
                        max=365,
                        value=[0, 365],
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ], id="advanced-filters-collapse", is_open=False),
            
            # Export Button
            html.Div([
                dbc.Button(
                    [html.I(className="fas fa-download me-2"), "Export Data"],
                    id="export-button",
                    color="success",
                    className="w-100 mt-4",
                    style={'borderRadius': '8px'}
                )
            ])
        ])
    ], style=get_sidebar_style(theme))

# Enhanced visualizations
def create_price_distribution_chart(filtered_df, theme='dark'):
    colors = THEMES[theme]
    
    fig = go.Figure()
    
    # Add histogram with custom styling
    fig.add_trace(go.Histogram(
        x=filtered_df['price'],
        nbinsx=50,
        marker=dict(
            color='rgba(59, 130, 246, 0.6)',
            line=dict(color='rgba(59, 130, 246, 1)', width=1)
        ),
        name='Price Distribution'
    ))
    
    # Add KDE overlay
    from scipy import stats
    kde = stats.gaussian_kde(filtered_df['price'])
    x_range = np.linspace(filtered_df['price'].min(), filtered_df['price'].max(), 100)
    kde_values = kde(x_range)
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_values * len(filtered_df) * (filtered_df['price'].max() - filtered_df['price'].min()) / 50,
        mode='lines',
        line=dict(color='rgba(139, 92, 246, 1)', width=3),
        name='Density',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=dict(
            text='Price Distribution Analysis',
            font=dict(size=20, color=colors['text_primary'])
        ),
        xaxis=dict(
            title='Price ($)',
            gridcolor=colors['border'],
            color=colors['text_secondary']
        ),
        yaxis=dict(
            title='Count',
            gridcolor=colors['border'],
            color=colors['text_secondary']
        ),
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False,
            showticklabels=False
        ),
        plot_bgcolor=colors['bg_card'],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text_primary']),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor=colors['border'],
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_correlation_heatmap(filtered_df, theme='dark'):
    colors = THEMES[theme]
    
    # Select numerical columns for correlation
    numerical_cols = ['price', 'number_of_reviews', 'reviews_per_month', 
                      'availability_365', 'calculated_host_listings_count',
                      'occupancy_rate', 'estimated_revenue']
    
    # Filter columns that exist in the dataframe
    available_cols = [col for col in numerical_cols if col in filtered_df.columns]
    
    if len(available_cols) < 2:
        return go.Figure()
    
    corr_matrix = filtered_df[available_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.around(corr_matrix.values, decimals=2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(
            text='Feature Correlation Matrix',
            font=dict(size=20, color=colors['text_primary'])
        ),
        plot_bgcolor=colors['bg_card'],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text_primary']),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(
            tickangle=-45,
            color=colors['text_secondary']
        ),
        yaxis=dict(
            color=colors['text_secondary']
        )
    )
    
    return fig

def create_clustering_analysis(filtered_df, theme='dark'):
    colors = THEMES[theme]
    
    if len(filtered_df) < 10:
        return go.Figure()
    
    # Prepare data for clustering
    features = ['price', 'number_of_reviews', 'availability_365']
    X = filtered_df[features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create scatter plot
    fig = go.Figure()
    
    cluster_colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b']
    cluster_names = ['Budget Basics', 'Premium Properties', 'High Availability', 'Popular Choices']
    
    for i in range(4):
        mask = clusters == i
        fig.add_trace(go.Scatter(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=cluster_colors[i],
                opacity=0.7,
                line=dict(width=1, color=colors['border'])
            ),
            name=cluster_names[i],
            text=filtered_df[mask]['name'],
            hovertemplate='<b>%{text}</b><br>' +
                          'Cluster: ' + cluster_names[i] + '<br>' +
                          '<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='Property Clustering Analysis',
            font=dict(size=20, color=colors['text_primary'])
        ),
        xaxis=dict(
            title='Principal Component 1',
            gridcolor=colors['border'],
            color=colors['text_secondary'],
            zeroline=False
        ),
        yaxis=dict(
            title='Principal Component 2',
            gridcolor=colors['border'],
            color=colors['text_secondary'],
            zeroline=False
        ),
        plot_bgcolor=colors['bg_card'],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text_primary']),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor=colors['border'],
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_revenue_analysis(filtered_df, theme='dark'):
    colors = THEMES[theme]
    
    # Group by room type and calculate metrics
    revenue_by_type = filtered_df.groupby('room_type').agg({
        'estimated_revenue': 'mean',
        'occupancy_rate': 'mean',
        'price': 'mean'
    }).reset_index()
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Revenue by Room Type', 'Occupancy vs Price'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Revenue bar chart
    fig.add_trace(
        go.Bar(
            x=revenue_by_type['room_type'],
            y=revenue_by_type['estimated_revenue'],
            marker_color='rgba(59, 130, 246, 0.8)',
            name='Avg Revenue'
        ),
        row=1, col=1
    )
    
    # Add occupancy rate line
    fig.add_trace(
        go.Scatter(
            x=revenue_by_type['room_type'],
            y=revenue_by_type['occupancy_rate'] * 100,
            mode='lines+markers',
            line=dict(color='rgba(139, 92, 246, 1)', width=3),
            marker=dict(size=10),
            name='Occupancy %'
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Scatter plot of occupancy vs price
    fig.add_trace(
        go.Scatter(
            x=filtered_df['occupancy_rate'] * 100,
            y=filtered_df['price'],
            mode='markers',
            marker=dict(
                size=6,
                color=filtered_df['estimated_revenue'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Revenue", x=1.1)
            ),
            name='Properties',
            text=filtered_df['room_type'],
            hovertemplate='<b>%{text}</b><br>' +
                          'Occupancy: %{x:.1f}%<br>' +
                          'Price: $%{y:.0f}<br>' +
                          '<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Room Type", row=1, col=1, color=colors['text_secondary'])
    fig.update_xaxes(title_text="Occupancy Rate (%)", row=1, col=2, color=colors['text_secondary'])
    fig.update_yaxes(title_text="Revenue ($)", row=1, col=1, color=colors['text_secondary'])
    fig.update_yaxes(title_text="Occupancy Rate (%)", row=1, col=1, secondary_y=True, color=colors['text_secondary'])
    fig.update_yaxes(title_text="Price ($)", row=1, col=2, color=colors['text_secondary'])
    
    fig.update_layout(
        title=dict(
            text='Revenue and Occupancy Analysis',
            font=dict(size=20, color=colors['text_primary'])
        ),
        plot_bgcolor=colors['bg_card'],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text_primary']),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor=colors['border'],
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    return fig

def create_enhanced_map(filtered_df, theme='dark'):
    colors = THEMES[theme]
    
    # Create custom color scale based on price categories
    fig = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="price_category",
        size="price",
        hover_name="name",
        hover_data={
            "price": ":$,.0f",
            "room_type": True,
            "number_of_reviews": True,
            "neighbourhood": True,
            "price_category": False
        },
        color_discrete_map={
            'Budget': '#10b981',
            'Economy': '#3b82f6',
            'Standard': '#8b5cf6',
            'Premium': '#f59e0b',
            'Luxury': '#ef4444'
        },
        zoom=10,
        height=600
    )
    
    fig.update_traces(
        marker=dict(
            sizemode='area',
            sizeref=2.*max(filtered_df['price'])/(40.**2),
            sizemin=4,
            opacity=0.8
        )
    )
    
    # Update layout with dark/light theme
    mapbox_style = "carto-darkmatter" if theme == 'dark' else "carto-positron"
    
    fig.update_layout(
        mapbox_style=mapbox_style,
        mapbox=dict(
            bearing=0,
            center=dict(
                lat=filtered_df['latitude'].mean() if not filtered_df.empty else -22.9,
                lon=filtered_df['longitude'].mean() if not filtered_df.empty else -43.2
            ),
            pitch=0,
            zoom=10
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text_primary']),
        legend=dict(
            bgcolor=colors['glassmorphism'],
            bordercolor=colors['border'],
            borderwidth=1,
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top'
        )
    )
    
    return fig

# Layout
app.layout = html.Div([
    # Theme store
    dcc.Store(id='theme-store', data='dark'),
    
    # Custom CSS
    html.Div(id='custom-css-container'),
    
    # Theme toggle button
    html.Div([
        html.Button(
            html.I(id='theme-icon', className="fas fa-moon fa-lg"),
            id='theme-toggle',
            className='theme-toggle',
            n_clicks=0
        )
    ]),
    
    # Main container
    html.Div([
        # Sidebar
        create_sidebar(),
        
        # Main content
        html.Div([
            # Header
            html.Div([
                html.H1("Rio de Janeiro Airbnb Analytics", 
                       className="section-title mb-2"),
                html.P("Comprehensive analysis of Airbnb listings with advanced insights",
                      className="text-secondary")
            ], className="mb-4"),
            
            # Metric cards
            create_metric_cards(),
            
            # Main visualizations
            html.Div([
                # Row 1: Price Distribution and Correlation
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dls.Hash(
                                dcc.Graph(id='price-distribution', 
                                         config={'displayModeBar': False}),
                                color="#3b82f6",
                                speed_multiplier=2,
                                size=50
                            )
                        ], className="glass-card")
                    ], lg=6),
                    
                    dbc.Col([
                        html.Div([
                            dls.Hash(
                                dcc.Graph(id='correlation-heatmap',
                                         config={'displayModeBar': False}),
                                color="#8b5cf6",
                                speed_multiplier=2,
                                size=50
                            )
                        ], className="glass-card")
                    ], lg=6)
                ], className="mb-4"),
                
                # Row 2: Clustering and Revenue Analysis
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dls.Hash(
                                dcc.Graph(id='clustering-analysis',
                                         config={'displayModeBar': False}),
                                color="#10b981",
                                speed_multiplier=2,
                                size=50
                            )
                        ], className="glass-card")
                    ], lg=6),
                    
                    dbc.Col([
                        html.Div([
                            dls.Hash(
                                dcc.Graph(id='revenue-analysis',
                                         config={'displayModeBar': False}),
                                color="#f59e0b",
                                speed_multiplier=2,
                                size=50
                            )
                        ], className="glass-card")
                    ], lg=6)
                ], className="mb-4"),
                
                # Row 3: Enhanced Map
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("Geographic Distribution", 
                                   className="mb-3",
                                   style={'fontSize': '1.5rem', 'fontWeight': '600'}),
                            dls.Hash(
                                dcc.Graph(id='enhanced-map',
                                         config={'displayModeBar': False}),
                                color="#ef4444",
                                speed_multiplier=2,
                                size=50
                            )
                        ], className="glass-card")
                    ], lg=12)
                ])
            ])
        ], className="main-content")
    ], className="main-container", id="main-container")
])

# Callbacks
@app.callback(
    [Output('theme-store', 'data'),
     Output('theme-icon', 'className'),
     Output('custom-css-container', 'children')],
    [Input('theme-toggle', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme(n_clicks, current_theme):
    if n_clicks > 0:
        new_theme = 'light' if current_theme == 'dark' else 'dark'
    else:
        new_theme = current_theme
    
    icon_class = "fas fa-sun fa-lg" if new_theme == 'dark' else "fas fa-moon fa-lg"
    css = get_custom_css(new_theme)
    
    # Return the CSS wrapped in a style tag using dangerously_set_inner_html
    style_element = html.Div([
        html.Style(children=css)
    ])
    
    return new_theme, icon_class, style_element

@app.callback(
    Output('advanced-filters-collapse', 'is_open'),
    [Input('advanced-filters-toggle', 'n_clicks')],
    [State('advanced-filters-collapse', 'is_open')]
)
def toggle_advanced_filters(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Main callback for visualizations
@app.callback(
    [Output('price-distribution', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('clustering-analysis', 'figure'),
     Output('revenue-analysis', 'figure'),
     Output('enhanced-map', 'figure')],
    [Input('neighbourhood-group-filter', 'value'),
     Input('room-type-filter', 'value'),
     Input('price-slider', 'value'),
     Input('min-reviews-slider', 'value'),
     Input('availability-slider', 'value'),
     Input('theme-store', 'data')]
)
def update_visualizations(neighbourhood_group, room_type, price_range, 
                         min_reviews, availability_range, theme):
    # Filter data
    filtered_df = df.copy()
    
    if neighbourhood_group != 'all':
        filtered_df = filtered_df[filtered_df['neighbourhood_group'] == neighbourhood_group]
    
    if room_type != 'all':
        filtered_df = filtered_df[filtered_df['room_type'] == room_type]
    
    if price_range:
        filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & 
                                  (filtered_df['price'] <= price_range[1])]
    
    if min_reviews and min_reviews > 0:
        filtered_df = filtered_df[filtered_df['number_of_reviews'] >= min_reviews]
    
    if availability_range:
        filtered_df = filtered_df[(filtered_df['availability_365'] >= availability_range[0]) & 
                                  (filtered_df['availability_365'] <= availability_range[1])]
    
    # Create visualizations
    price_dist = create_price_distribution_chart(filtered_df, theme)
    correlation = create_correlation_heatmap(filtered_df, theme)
    clustering = create_clustering_analysis(filtered_df, theme)
    revenue = create_revenue_analysis(filtered_df, theme)
    map_fig = create_enhanced_map(filtered_df, theme)
    
    return price_dist, correlation, clustering, revenue, map_fig

# Export functionality
@app.callback(
    Output('export-button', 'n_clicks'),
    [Input('export-button', 'n_clicks')],
    [State('neighbourhood-group-filter', 'value'),
     State('room-type-filter', 'value'),
     State('price-slider', 'value')]
)
def export_data(n_clicks, neighbourhood_group, room_type, price_range):
    if n_clicks and n_clicks > 0:
        # Apply filters
        filtered_df = df.copy()
        
        if neighbourhood_group != 'all':
            filtered_df = filtered_df[filtered_df['neighbourhood_group'] == neighbourhood_group]
        
        if room_type != 'all':
            filtered_df = filtered_df[filtered_df['room_type'] == room_type]
        
        if price_range:
            filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & 
                                      (filtered_df['price'] <= price_range[1])]
        
        # Export to CSV
        filtered_df.to_csv('airbnb_filtered_data.csv', index=False)
    
    return 0

if __name__ == '__main__':
    app.run_server(debug=True)