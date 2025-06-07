import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Load and process data
def load_data():
    try:
        df = pd.read_csv('listings.csv')
        
        # Handle price column - check if it's already numeric or string
        if df['price'].dtype == 'object':
            df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float, errors='ignore')
        else:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        df = df[df['price'].notna() & (df['price'] > 0) & (df['price'] < 10000)]
        
        # Feature engineering
        df['occupancy_rate'] = (365 - df['availability_365']) / 365
        df['estimated_revenue'] = df['price'] * (365 - df['availability_365'])
        df['price_category'] = pd.qcut(df['price'], q=5, labels=['Budget', 'Economy', 'Standard', 'Premium', 'Luxury'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_data()

# Define the app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏠 Rio de Janeiro Airbnb Analytics", 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'marginBottom': '30px',
                    'fontSize': '3rem',
                    'fontWeight': 'bold'
                })
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px'}),
    
    # Filters
    html.Div([
        html.Div([
            html.Label("Neighborhood Group:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='neighborhood-filter',
                options=[{'label': 'All', 'value': 'all'}] + 
                        [{'label': ng, 'value': ng} for ng in df['neighbourhood_group'].unique() if pd.notna(ng)],
                value='all',
                style={'marginBottom': '20px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}),
        
        html.Div([
            html.Label("Room Type:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='room-type-filter',
                options=[{'label': 'All', 'value': 'all'}] + 
                        [{'label': rt, 'value': rt} for rt in df['room_type'].unique() if pd.notna(rt)],
                value='all',
                style={'marginBottom': '20px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}),
        
        html.Div([
            html.Label("Price Range:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.RangeSlider(
                id='price-slider',
                min=df['price'].min() if not df.empty else 0,
                max=df['price'].quantile(0.95) if not df.empty else 1000,
                value=[df['price'].min() if not df.empty else 0, 
                       df['price'].quantile(0.75) if not df.empty else 500],
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
                step=10
            )
        ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'margin': '20px 0'}),
    
    # Key Metrics
    html.Div(id='metrics-row', style={'margin': '20px 0'}),
    
    # Charts
    html.Div([
        # Row 1: Price Distribution and Room Type Analysis
        html.Div([
            html.Div([
                dcc.Graph(id='price-distribution')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='room-type-analysis')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        # Row 2: Map and Revenue Analysis
        html.Div([
            html.Div([
                dcc.Graph(id='map-visualization')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='revenue-analysis')
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    ])
], style={'fontFamily': 'Arial, sans-serif', 'margin': '0', 'padding': '0'})

# Callback for updating all charts
@app.callback(
    [Output('metrics-row', 'children'),
     Output('price-distribution', 'figure'),
     Output('room-type-analysis', 'figure'),
     Output('map-visualization', 'figure'),
     Output('revenue-analysis', 'figure')],
    [Input('neighborhood-filter', 'value'),
     Input('room-type-filter', 'value'),
     Input('price-slider', 'value')]
)
def update_dashboard(neighborhood, room_type, price_range):
    # Filter data
    filtered_df = df.copy()
    
    if neighborhood != 'all':
        filtered_df = filtered_df[filtered_df['neighbourhood_group'] == neighborhood]
    
    if room_type != 'all':
        filtered_df = filtered_df[filtered_df['room_type'] == room_type]
    
    if price_range:
        filtered_df = filtered_df[
            (filtered_df['price'] >= price_range[0]) & 
            (filtered_df['price'] <= price_range[1])
        ]
    
    # Create metrics
    total_listings = len(filtered_df)
    avg_price = filtered_df['price'].mean() if not filtered_df.empty else 0
    avg_occupancy = filtered_df['occupancy_rate'].mean() * 100 if not filtered_df.empty else 0
    total_revenue = filtered_df['estimated_revenue'].sum() if not filtered_df.empty else 0
    
    metrics = html.Div([
        html.Div([
            html.H3(f"{total_listings:,}", style={'color': '#3498db', 'margin': '0', 'fontSize': '2.5rem'}),
            html.P("Total Listings", style={'margin': '5px 0', 'fontWeight': 'bold'})
        ], style={
            'textAlign': 'center', 
            'backgroundColor': 'white', 
            'padding': '20px', 
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'width': '22%',
            'display': 'inline-block',
            'margin': '0 1.5%'
        }),
        
        html.Div([
            html.H3(f"${avg_price:.0f}", style={'color': '#e74c3c', 'margin': '0', 'fontSize': '2.5rem'}),
            html.P("Average Price", style={'margin': '5px 0', 'fontWeight': 'bold'})
        ], style={
            'textAlign': 'center', 
            'backgroundColor': 'white', 
            'padding': '20px', 
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'width': '22%',
            'display': 'inline-block',
            'margin': '0 1.5%'
        }),
        
        html.Div([
            html.H3(f"{avg_occupancy:.1f}%", style={'color': '#2ecc71', 'margin': '0', 'fontSize': '2.5rem'}),
            html.P("Avg Occupancy", style={'margin': '5px 0', 'fontWeight': 'bold'})
        ], style={
            'textAlign': 'center', 
            'backgroundColor': 'white', 
            'padding': '20px', 
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'width': '22%',
            'display': 'inline-block',
            'margin': '0 1.5%'
        }),
        
        html.Div([
            html.H3(f"${total_revenue/1000000:.1f}M", style={'color': '#f39c12', 'margin': '0', 'fontSize': '2.5rem'}),
            html.P("Total Revenue", style={'margin': '5px 0', 'fontWeight': 'bold'})
        ], style={
            'textAlign': 'center', 
            'backgroundColor': 'white', 
            'padding': '20px', 
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'width': '22%',
            'display': 'inline-block',
            'margin': '0 1.5%'
        })
    ])
    
    # Create visualizations
    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available for selected filters", 
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        empty_fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
        return metrics, empty_fig, empty_fig, empty_fig, empty_fig
    
    # 1. Price Distribution
    fig_price = go.Figure()
    fig_price.add_trace(go.Histogram(
        x=filtered_df['price'],
        nbinsx=30,
        marker_color='rgba(52, 152, 219, 0.7)',
        name='Price Distribution'
    ))
    fig_price.update_layout(
        title='Price Distribution',
        xaxis_title='Price ($)',
        yaxis_title='Count',
        template='plotly_white',
        height=400
    )
    
    # 2. Room Type Analysis
    room_type_counts = filtered_df['room_type'].value_counts()
    fig_room = go.Figure(data=[go.Pie(
        labels=room_type_counts.index,
        values=room_type_counts.values,
        hole=0.4,
        marker_colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    )])
    fig_room.update_layout(
        title='Distribution by Room Type',
        height=400
    )
    
    # 3. Map Visualization
    sample_size = min(1000, len(filtered_df))  # Limit for performance
    map_df = filtered_df.sample(sample_size) if len(filtered_df) > sample_size else filtered_df
    
    fig_map = px.scatter_mapbox(
        map_df,
        lat='latitude',
        lon='longitude',
        color='price_category',
        size='price',
        hover_name='name',
        hover_data=['price', 'room_type', 'number_of_reviews'],
        color_discrete_map={
            'Budget': '#2ecc71',
            'Economy': '#3498db',
            'Standard': '#f39c12',
            'Premium': '#e67e22',
            'Luxury': '#e74c3c'
        },
        mapbox_style='carto-positron',
        height=400,
        zoom=10
    )
    
    fig_map.update_layout(
        title='Geographic Distribution',
        mapbox=dict(
            center=dict(
                lat=filtered_df['latitude'].mean(),
                lon=filtered_df['longitude'].mean()
            )
        )
    )
    
    # 4. Revenue Analysis
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
            marker_color='rgba(52, 152, 219, 0.8)'
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
            line=dict(color='rgba(231, 76, 60, 1)', width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    fig_revenue.update_xaxes(title_text="Room Type")
    fig_revenue.update_yaxes(title_text="Average Revenue ($)", secondary_y=False)
    fig_revenue.update_yaxes(title_text="Occupancy Rate (%)", secondary_y=True)
    fig_revenue.update_layout(
        title='Revenue & Occupancy Analysis',
        height=400,
        template='plotly_white'
    )
    
    return metrics, fig_price, fig_room, fig_map, fig_revenue

if __name__ == '__main__':
    print("🚀 Starting Dash app...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    app.run(debug=True)