import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
from datetime import datetime

# Load data from listings.csv
def load_data():
    if not os.path.exists('listings.csv'):
        raise FileNotFoundError("listings.csv file not found in the current directory")
    
    df = pd.read_csv('listings.csv', 
                    parse_dates=['last_review'],
                    dtype={
                        'id': 'int64',
                        'host_id': 'int64',
                        'price': 'float64',
                        'minimum_nights': 'int64',
                        'number_of_reviews': 'int64',
                        'reviews_per_month': 'float64',
                        'calculated_host_listings_count': 'int64',
                        'availability_365': 'int64',
                        'number_of_reviews_ltm': 'int64'
                    })
    
    # Clean data
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price', 'latitude', 'longitude'])
    df['price'] = df['price'].round(2)
    
    # Fill missing values
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['neighbourhood_group'] = df['neighbourhood_group'].fillna('Unknown')
    df['neighbourhood'] = df['neighbourhood'].fillna('Unknown')
    
    # Create price categories for map filtering
    df['price_category'] = pd.cut(df['price'],
                                 bins=[0, 100, 200, 300, 400, 500, float('inf')],
                                 labels=['<R$100', 'R$100-200', 'R$200-300', 
                                         'R$300-400', 'R$400-500', '>R$500'])
    
    return df

try:
    df = load_data()
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame({
        'neighbourhood_group': ['Unknown'],
        'neighbourhood': ['Unknown'],
        'room_type': ['Unknown'],
        'price': [0],
        'price_category': ['<R$100'],
        'minimum_nights': [0],
        'number_of_reviews': [0],
        'last_review': pd.to_datetime(['2023-01-01']),
        'reviews_per_month': [0],
        'latitude': [-22.9068],
        'longitude': [-43.1729]
    })

# Initialize the Dash app
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
]

app = dash.Dash(__name__, 
                title="Airbnb Rio Analytics Dashboard",
                external_stylesheets=external_stylesheets)
server = app.server

# Custom CSS styles
styles = {
    'app': {
        'fontFamily': 'Roboto, sans-serif',
        'backgroundColor': '#f8fafc',
        'minHeight': '100vh'
    },
    'header': {
        'background': 'linear-gradient(135deg, #4f46e5, #7c3aed)',
        'color': 'white',
        'padding': '2rem',
        'marginBottom': '2rem',
        'textAlign': 'center',
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
    },
    'headerTitle': {
        'fontSize': '2.5rem',
        'marginBottom': '0.5rem',
        'fontWeight': '700'
    },
    'headerDescription': {
        'opacity': '0.9',
        'fontSize': '1.2rem',
        'fontWeight': '300'
    },
    'container': {
        'maxWidth': '1600px',
        'margin': '0 auto',
        'padding': '0 1.5rem'
    },
    'filters': {
        'backgroundColor': 'white',
        'padding': '1.5rem',
        'borderRadius': '12px',
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.05)',
        'marginBottom': '1.5rem'
    },
    'filterLabel': {
        'display': 'block',
        'marginBottom': '0.5rem',
        'fontWeight': '500',
        'color': '#4b5563'
    },
    'dropdown': {
        'marginBottom': '1.5rem',
        'fontFamily': 'Roboto, sans-serif'
    },
    'rangeSlider': {
        'marginTop': '0.5rem'
    },
    'chartsGrid': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(2, 1fr)',
        'gap': '1.5rem'
    },
    'card': {
        'backgroundColor': 'white',
        'borderRadius': '12px',
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.05)',
        'padding': '1.5rem',
        'transition': 'all 0.3s ease',
        'border': '1px solid #f3f4f6'
    },
    'wideCard': {
        'gridColumn': 'span 2',
        'backgroundColor': 'white',
        'borderRadius': '12px',
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.05)',
        'padding': '1.5rem',
        'transition': 'all 0.3s ease',
        'border': '1px solid #f3f4f6'
    },
    'mapControls': {
        'backgroundColor': '#f9fafb',
        'padding': '1rem',
        'borderRadius': '8px',
        'marginBottom': '1rem',
        'border': '1px solid #e5e7eb'
    }
}

# Define the app layout with enhanced map filters
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1("Airbnb Rio Analytics Dashboard", 
                    style=styles['headerTitle']),
            html.P("Explore pricing trends and property comparisons in Rio de Janeiro",
                  style=styles['headerDescription']),
            html.Div([
                html.I(className="fas fa-home", style={'marginRight': '8px'}),
                "thiago thomas, ..."
            ], style={'marginTop': '1rem', 'opacity': '0.7', 'fontSize': '0.9rem'})
        ], style=styles['header']),
        
        html.Div([
            # Filters
            html.Div([
                html.Label("Neighborhood Group:", style=styles['filterLabel']),
                dcc.Dropdown(
                    id='neighborhood-group-filter',
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': n, 'value': n} for n in sorted(df['neighbourhood_group'].unique()) if pd.notna(n)],
                    value='All',
                    clearable=False,
                    style=styles['dropdown'],
                    placeholder="Select neighborhood group..."
                ),
                
                html.Label("Neighborhood:", style=styles['filterLabel']),
                dcc.Dropdown(
                    id='neighborhood-filter',
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': n, 'value': n} for n in sorted(df['neighbourhood'].unique()) if pd.notna(n)],
                    value='All',
                    clearable=False,
                    style=styles['dropdown'],
                    placeholder="Select neighborhood..."
                ),
                
                html.Label("Room Type:", style=styles['filterLabel']),
                dcc.Dropdown(
                    id='room-type-filter',
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': t, 'value': t} for t in sorted(df['room_type'].unique()) if pd.notna(t)],
                    value='All',
                    clearable=False,
                    style=styles['dropdown'],
                    placeholder="Select room type..."
                ),
                
                html.Label(f"Price Range: (R${int(df['price'].min())} - R${int(df['price'].max())})", 
                          style=styles['filterLabel']),
                dcc.RangeSlider(
                    id='price-range',
                    min=int(df['price'].min()),
                    max=int(df['price'].max()),
                    step=10,
                    value=[int(df['price'].quantile(0.1)), int(df['price'].quantile(0.9))],
                    marks={
                        int(df['price'].min()): {'label': f'R${int(df["price"].min())}', 'style': {'color': '#4b5563'}},
                        int(df['price'].max()): {'label': f'R${int(df["price"].max())}', 'style': {'color': '#4b5563'}}
                    },
                    tooltip={'placement': 'bottom', 'always_visible': True}
                ),
            ], style=styles['filters']),
            
            # Main charts
            html.Div([
                html.Div([
                    dcc.Graph(id='price-distribution'),
                ], className="card", style=styles['card']),
                
                html.Div([
                    dcc.Graph(id='room-type-pie'),
                ], className="card", style=styles['card']),
                
                html.Div([
                    dcc.Graph(id='price-comparison'),
                ], className="card", style=styles['card']),
                
                html.Div([
                    dcc.Graph(id='amenity-impact'),
                ], className="card", style=styles['card']),
                
                html.Div([
                    dcc.Graph(id='price-vs-reviews'),
                ], className="card", style=styles['card']),
                
                html.Div([
                    dcc.Graph(id='price-trends'),
                ], className="card", style=styles['card']),
                
                # Enhanced Map Visualization with its own controls
                html.Div([
                    html.Div([
                        html.Label("Map Display Options:", style=styles['filterLabel']),
                        dcc.RadioItems(
                            id='map-color-by',
                            options=[
                                {'label': 'Color by Room Type', 'value': 'room_type'},
                                {'label': 'Color by Price Category', 'value': 'price_category'}
                            ],
                            value='room_type',
                            inline=True,
                            style={'marginBottom': '15px'}
                        ),
                        dcc.RadioItems(
                            id='map-size-by',
                            options=[
                                {'label': 'Size by Price', 'value': 'price'},
                                {'label': 'Size by Number of Reviews', 'value': 'number_of_reviews'}
                            ],
                            value='price',
                            inline=True
                        ),
                        
                        html.Label("Filter by Price Category:", style={**styles['filterLabel'], 'marginTop': '15px'}),
                        dcc.Dropdown(
                            id='map-price-category',
                            options=[{'label': 'All', 'value': 'All'}] + 
                                    [{'label': cat, 'value': cat} for cat in sorted(df['price_category'].unique())],
                            value='All',
                            clearable=False,
                            style={**styles['dropdown'], 'marginBottom': '0'}
                        ),
                    ], style=styles['mapControls']),
                    
                    dcc.Graph(id='geo-map'),
                ], className="card", style=styles['wideCard']),
            ], style=styles['chartsGrid']),
        ], style=styles['container']),
    ], style=styles['app'])
])

# Callback to update all visualizations
@app.callback(
    Output('price-distribution', 'figure'),
    Output('room-type-pie', 'figure'),
    Output('price-comparison', 'figure'),
    Output('amenity-impact', 'figure'),
    Output('price-vs-reviews', 'figure'),
    Output('price-trends', 'figure'),
    Output('geo-map', 'figure'),
    Input('neighborhood-group-filter', 'value'),
    Input('neighborhood-filter', 'value'),
    Input('room-type-filter', 'value'),
    Input('price-range', 'value'),
    Input('map-price-category', 'value'),
    Input('map-color-by', 'value'),
    Input('map-size-by', 'value')
)
def update_charts(neighborhood_group, neighborhood, room_type, price_range, 
                 map_price_category, map_color_by, map_size_by):
    # Filter data based on main filters
    filtered_df = df.copy()
    
    if neighborhood_group != 'All':
        filtered_df = filtered_df[filtered_df['neighbourhood_group'] == neighborhood_group]
    
    if neighborhood != 'All':
        filtered_df = filtered_df[filtered_df['neighbourhood'] == neighborhood]
    
    if room_type != 'All':
        filtered_df = filtered_df[filtered_df['room_type'] == room_type]
    
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_range[0]) & 
        (filtered_df['price'] <= price_range[1])
    ]
    
    # Apply additional map-specific price category filter
    if map_price_category != 'All':
        filtered_df = filtered_df[filtered_df['price_category'] == map_price_category]
    
    # Color scheme
    colors = {
        'primary': '#4f46e5',
        'secondary': '#7c3aed',
        'accent': '#10b981',
        'text': '#1f2937',
        'background': '#f8fafc'
    }
    
    # 1. Price Distribution Histogram
    price_hist = px.histogram(
        filtered_df, 
        x='price',
        nbins=30,
        title='Price Distribution',
        labels={'price': 'Price (R$)'},
        color_discrete_sequence=[colors['primary']],
        template='plotly_white'
    )
    price_hist.update_layout(
        bargap=0.1,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        title_font=dict(size=18, color=colors['text']),
        font=dict(family='Roboto')
    )
    
    # 2. Room Type Pie Chart
    room_counts = filtered_df['room_type'].value_counts().reset_index()
    room_counts.columns = ['room_type', 'count']
    
    room_pie = px.pie(
        room_counts,
        names='room_type',
        values='count',
        title='Room Type Distribution',
        hole=0.4,
        color_discrete_sequence=[colors['primary'], colors['secondary'], colors['accent'], '#f59e0b'],
        template='plotly_white'
    )
    room_pie.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(line=dict(color='white', width=1))
    )
    room_pie.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
        title_font=dict(size=18, color=colors['text']),
        font=dict(family='Roboto')
    )
    
    # 3. Price Comparison by Property Type
    price_comparison = px.box(
        filtered_df,
        x='room_type',
        y='price',
        title='Price Comparison by Room Type',
        labels={'price': 'Price (R$)', 'room_type': 'Room Type'},
        color='room_type',
        color_discrete_sequence=[colors['primary'], colors['secondary'], colors['accent'], '#f59e0b'],
        template='plotly_white'
    )
    price_comparison.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title=None,
        showlegend=False,
        title_font=dict(size=18, color=colors['text']),
        font=dict(family='Roboto')
    )
    
    # 4. Amenity Impact on Price (using host listings count as proxy)
    amenity_impact = px.box(
        filtered_df,
        x='room_type',
        y='price',
        color='calculated_host_listings_count',
        title='Price by Host Listings Count',
        labels={'price': 'Price (R$)', 'room_type': 'Room Type', 'calculated_host_listings_count': 'Host Listings'},
        template='plotly_white'
    )
    amenity_impact.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text='Host Listings',
        title_font=dict(size=18, color=colors['text']),
        font=dict(family='Roboto')
    )
    
    # 5. Price vs. Reviews Scatter Plot
    price_vs_reviews = px.scatter(
        filtered_df,
        x='number_of_reviews',
        y='price',
        color='room_type',
        title='Price vs. Number of Reviews',
        labels={'price': 'Price (R$)', 'number_of_reviews': 'Number of Reviews'},
        opacity=0.7,
        color_discrete_sequence=[colors['primary'], colors['secondary'], colors['accent'], '#f59e0b'],
        template='plotly_white'
    )
    price_vs_reviews.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text='Room Type',
        title_font=dict(size=18, color=colors['text']),
        font=dict(family='Roboto')
    )
    
    # 6. Fixed Price Trends Over Time
    # Create a proper date sequence for the trends
    if not filtered_df.empty:
        # Create a date range for the last 12 months
        end_date = filtered_df['last_review'].max()
        start_date = end_date - pd.DateOffset(months=11)
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Group by month and room type
        monthly_avg = filtered_df.groupby([
            pd.Grouper(key='last_review', freq='MS'),
            'room_type'
        ])['price'].mean().reset_index()
        
        # Ensure all months are represented
        all_months = pd.DataFrame({
            'last_review': date_range
        })
        all_room_types = filtered_df['room_type'].unique()
        
        # Create a complete grid of all months and room types
        complete_grid = pd.MultiIndex.from_product(
            [date_range, all_room_types],
            names=['last_review', 'room_type']
        ).to_frame(index=False)
        
        # Merge with actual data
        monthly_avg = pd.merge(complete_grid, monthly_avg, 
                              on=['last_review', 'room_type'], 
                              how='left')
        
        # Format month names
        monthly_avg['month'] = monthly_avg['last_review'].dt.strftime('%B %Y')
        
        # Order months chronologically
        month_order = monthly_avg['last_review'].dt.strftime('%B %Y').unique()
        monthly_avg['month'] = pd.Categorical(monthly_avg['month'], categories=month_order, ordered=True)
    else:
        monthly_avg = pd.DataFrame(columns=['month', 'room_type', 'price'])
    
    price_trends = px.line(
        monthly_avg,
        x='month',
        y='price',
        color='room_type',
        title='Monthly Average Price by Room Type',
        labels={'price': 'Average Price (R$)', 'month': 'Month'},
        color_discrete_sequence=[colors['primary'], colors['secondary'], colors['accent'], '#f59e0b'],
        template='plotly_white'
    )
    price_trends.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text='Room Type',
        title_font=dict(size=18, color=colors['text']),
        font=dict(family='Roboto')
    )
    price_trends.update_traces(line=dict(width=3))
    
    # 7. Enhanced Geographic Map Visualization with dynamic controls
    if not filtered_df.empty:
        # Determine color and size based on controls
        color_column = map_color_by
        size_column = map_size_by
        
        # Create custom hover data
        hover_data = {
            'name': True,
            'neighbourhood': True,
            'price': ':.2f',
            'number_of_reviews': True,
            'room_type': True,
            'price_category': True
        }
        
        # Adjust size scale based on what we're sizing by
        size_max = 20 if map_size_by == 'price' else 30
        
        # Use scatter_map instead of scatter_mapbox to avoid deprecation warning
        geo_map = px.scatter_mapbox(
            filtered_df,
            lat='latitude',
            lon='longitude',
            color=color_column,
            size=size_column,
            hover_name='name',
            hover_data=hover_data,
            title='Airbnb Listings Map',
            color_discrete_sequence=[colors['primary'], colors['secondary'], 
                                   colors['accent'], '#f59e0b', '#6366f1', '#8b5cf6'],
            zoom=11,
            height=600,
            size_max=size_max
        )
        
        # Set map style
        geo_map.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":40,"l":0,"b":0},
            title_font=dict(size=18, color=colors['text']),
            font=dict(family='Roboto'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Center the map on Rio de Janeiro
        geo_map.update_layout(
            mapbox=dict(
                center=dict(lat=-22.9068, lon=-43.1729)
            )
        )
    else:
        geo_map = px.scatter_mapbox(title='No data available for selected filters')
        geo_map.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":40,"l":0,"b":0},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    
    return price_hist, room_pie, price_comparison, amenity_impact, price_vs_reviews, price_trends, geo_map

# Run the app
if __name__ == '__main__':
    app.run(debug=True)