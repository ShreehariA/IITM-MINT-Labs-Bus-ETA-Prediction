"""
Extract actual bus stop locations from GPS data

This script analyzes GPS data to find the actual coordinates of bus stops
based on where buses frequently stop/slow down.

Usage:
    python extract_stops.py
"""

import pandas as pd
import numpy as np
from data_loader import load_single_day
from collections import Counter

# Load one day of data
print("Loading GPS data...")
df = load_single_day('20251103')

# Find where buses stop (speed < 2 km/h for > 30 seconds)
print("\nFinding stop locations...")
df = df.sort_values(['IMEI', 'DateTime'])

stops = []
for imei in df['IMEI'].unique():
    bus_data = df[df['IMEI'] == imei].copy()
    
    # Find stopped periods
    bus_data['is_stopped'] = bus_data['Speed'] < 2
    bus_data['stop_group'] = (bus_data['is_stopped'] != bus_data['is_stopped'].shift()).cumsum()
    
    for group_id, group in bus_data[bus_data['is_stopped']].groupby('stop_group'):
        if len(group) >= 30:  # Stopped for 30+ seconds
            # Get center of stop cluster
            lat = group['Latitude'].median()
            lon = group['Longitude'].median()
            duration = len(group)
            stops.append({'lat': lat, 'lon': lon, 'duration': duration})

stops_df = pd.DataFrame(stops)

# Cluster nearby stops (within 50 meters)
from sklearn.cluster import DBSCAN
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Convert to radians for clustering
coords = np.radians(stops_df[['lat', 'lon']].values)

# Cluster stops (eps=50m in radians ≈ 0.00045)
clustering = DBSCAN(eps=0.00045, min_samples=5, metric='haversine').fit(coords)
stops_df['cluster'] = clustering.labels_

# Get cluster centers
print("\nBus stop locations found:")
print("="*80)

bus_stops = {}
for cluster_id in sorted(stops_df[stops_df['cluster'] >= 0]['cluster'].unique()):
    cluster_data = stops_df[stops_df['cluster'] == cluster_id]
    lat = cluster_data['lat'].median()
    lon = cluster_data['lon'].median()
    count = len(cluster_data)
    
    bus_stops[cluster_id + 1] = {
        'lat': lat,
        'lon': lon,
        'count': count
    }
    
    print(f"Stop {cluster_id + 1}: lat={lat:.6f}, lon={lon:.6f} ({count} observations)")

# Generate Python code
print("\n" + "="*80)
print("Copy this into preprocessing_unified.py:")
print("="*80)
print("\nBUS_STOPS = {")
for stop_id, info in bus_stops.items():
    print(f"    {stop_id}: {{'name': 'Stop {stop_id}', 'lat': {info['lat']:.6f}, 'lon': {info['lon']:.6f}}},")
print("}")

print(f"\n✓ Found {len(bus_stops)} bus stops")
