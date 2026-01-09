
---

## 🔧 Data Preprocessing & Feature Engineering

The raw GPS data undergoes a comprehensive 9-step preprocessing pipeline to transform 17.2M GPS points into 4,208 training examples with 53 engineered features.

### Preprocessing Pipeline

#### Step 1: Data Cleaning & Validation
**Objective**: Filter invalid GPS points and ensure data quality

**Operations:**
- Remove (0,0) coordinates
- Apply geofencing (Chennai bounds: 12.85-13.15°N, 80.10-80.35°E)
- Filter out-of-bounds speeds (>150 km/h)
- Remove duplicate timestamps
- Validate GPS accuracy

**Result**: 17.2M → 4.7M valid points (27.3% retention)

---

#### Step 2: Trip Detection
**Objective**: Segment continuous GPS stream into individual bus trips

**Algorithm:**
- Identify gaps >15 minutes as trip boundaries
- Require minimum 20 points per trip
- Detect stationary periods (speed=0 for >30s)
- Handle overnight gaps

**Result**: 501 distinct trips identified

---

#### Step 3: Route Assignment
**Objective**: Match each trip to a known bus route

**Method:**
- Analyze stop visit sequence
- Compare with 10 predefined routes
- Use fuzzy matching for partial routes
- Fallback to nearest-neighbor for unmatched trips

**Result**: 266 trips matched (53.1% success rate)

---

#### Step 4: Distance Features
**Objective**: Calculate spatial relationships

**Features Derived:**
1. **`dist_to_stop_m`** - Haversine distance to target stop (meters)
2. **`dist_to_stop_euclidean`** - Euclidean distance (meters)
3. **`dist_to_stop_manhattan`** - Manhattan distance (meters)
4. **`dist_from_origin`** - Distance traveled from trip start
5. **`dist_remaining`** - Estimated distance to destination
6. **`nearest_stop_id`** - ID of closest bus stop
7. **`nearest_stop_dist`** - Distance to nearest stop

**Significance**: Distance-based features are the most predictive (correlation: 0.85 with ETA)

---

#### Step 5: Temporal Features
**Objective**: Capture time-of-day and day-of-week patterns

**Features Derived:**
1. **`hour`** - Hour of day (0-23)
2. **`day_of_week`** - Day (0=Monday, 6=Sunday)
3. **`is_weekend`** - Binary weekend indicator
4. **`is_rush_hour`** - Peak hours (7-10 AM, 4-7 PM)
5. **`hour_sin`** - Cyclical hour encoding (sin)
6. **`hour_cos`** - Cyclical hour encoding (cos)
7. **`time_since_midnight`** - Seconds since midnight

**Significance**: Rush hour adds ~150 min median delay (from EDA)

---

#### Step 6: Speed Dynamics
**Objective**: Capture movement patterns and traffic conditions

**Features Derived:**
1. **`speed_kmh`** - Current speed (km/h)
2. **`speed_mps`** - Current speed (m/s)
3. **`avg_speed_last_5`** - Rolling 5-point average speed
4. **`avg_speed_last_10`** - Rolling 10-point average speed
5. **`speed_variance`** - Speed variance (traffic indicator)
6. **`is_moving`** - Binary movement indicator (speed >1 km/h)
7. **`stationary_ratio`** - Fraction of time stationary
8. **`avg_stationary_duration`** - Average stop duration
9. **`acceleration`** - Speed change rate

**Significance**: 54% stationary time explains why simple physics models fail

---

#### Step 7: Derived Features
**Objective**: Create interaction and domain-specific features

**Features Derived:**
1. **`time_to_stop_naive`** - Distance / average_speed (baseline ETA)
2. **`speed_x_hour`** - Speed-hour interaction
3. **`speed_x_is_rush_hour`** - Speed-rush_hour interaction
4. **`dist_x_speed_variance`** - Distance-traffic interaction
5. **`stops_remaining`** - Number of stops to destination
6. **`stop_sequence`** - Current position in route
7. **`route_progress`** - Fraction of route completed (0-1)
8. **`route_id`** - Encoded route identifier
9. **`origin_stop`** - Trip starting stop
10. **`destination_stop`** - Trip ending stop

**Significance**: Interaction features capture non-linear patterns (e.g., slow speed at rush hour = high delay)

---

#### Step 8: Stop Arrival Detection
**Objective**: Identify when buses arrive at stops (for ground truth labels)

**Algorithm:**
- Detect when bus enters stop radius (<50m)
- Identify stationary period (speed <1 km/h for >30s)
- Record arrival timestamp
- Calculate dwell time

**Result**: 1,247 stop arrivals identified

---

#### Step 9: Training Data Generation
**Objective**: Create (features, label) pairs for model training

**Process:**
1. For each stop arrival, sample GPS points before it
2. Calculate actual ETA from sample time to arrival time
3. Extract all 53 features at sample point
4. Filter samples with ETA >30 minutes (outliers)
5. Balance dataset across routes and times

**Sampling Strategy:**
- 10-second intervals before each arrival
- Maximum 30 minutes lookback
- Minimum 30 seconds before arrival

**Result**: 4,208 training examples

---

### Complete Feature List (53 Features)

#### Spatial Features (7)
| Feature | Description | Type |
|---------|-------------|------|
| `dist_to_stop_m` | Haversine distance to target stop | Continuous |
| `dist_to_stop_euclidean` | Euclidean distance | Continuous |
| `dist_to_stop_manhattan` | Manhattan distance | Continuous |
| `dist_from_origin` | Distance from trip start | Continuous |
| `dist_remaining` | Distance to destination | Continuous |
| `nearest_stop_id` | Closest stop ID | Categorical |
| `nearest_stop_dist` | Distance to nearest stop | Continuous |

#### Temporal Features (7)
| Feature | Description | Type |
|---------|-------------|------|
| `hour` | Hour of day (0-23) | Continuous |
| `day_of_week` | Day (0-6) | Categorical |
| `is_weekend` | Weekend indicator | Binary |
| `is_rush_hour` | Peak hours indicator | Binary |
| `hour_sin` | Cyclical hour (sin) | Continuous |
| `hour_cos` | Cyclical hour (cos) | Continuous |
| `time_since_midnight` | Seconds since midnight | Continuous |

#### Speed & Movement Features (9)
| Feature | Description | Type |
|---------|-------------|------|
| `speed_kmh` | Current speed (km/h) | Continuous |
| `speed_mps` | Current speed (m/s) | Continuous |
| `avg_speed_last_5` | 5-point rolling average | Continuous |
| `avg_speed_last_10` | 10-point rolling average | Continuous |
| `speed_variance` | Speed variance | Continuous |
| `is_moving` | Movement indicator | Binary |
| `stationary_ratio` | Fraction stationary | Continuous |
| `avg_stationary_duration` | Average stop time | Continuous |
| `acceleration` | Speed change rate | Continuous |

#### Route & Position Features (10)
| Feature | Description | Type |
|---------|-------------|------|
| `route_id` | Route identifier | Categorical |
| `origin_stop` | Starting stop | Categorical |
| `destination_stop` | Ending stop | Categorical |
| `stops_remaining` | Stops to destination | Continuous |
| `stop_sequence` | Position in route | Continuous |
| `route_progress` | Completion fraction | Continuous |
| `target_stop_id` | Prediction target | Categorical |
| `current_stop_id` | Current/last stop | Categorical |
| `next_stop_id` | Next scheduled stop | Categorical |
| `prev_stop_id` | Previous stop | Categorical |

#### Interaction Features (10)
| Feature | Description | Type |
|---------|-------------|------|
| `time_to_stop_naive` | Distance / avg_speed | Continuous |
| `speed_x_hour` | Speed × hour | Continuous |
| `speed_x_is_rush_hour` | Speed × rush_hour | Continuous |
| `dist_x_speed_variance` | Distance × traffic | Continuous |
| `speed_x_dist` | Speed × distance | Continuous |
| `hour_x_route` | Hour × route | Continuous |
| `day_x_route` | Day × route | Continuous |
| `dist_x_stops_remaining` | Distance × stops | Continuous |
| `speed_x_route_progress` | Speed × progress | Continuous |
| `is_rush_hour_x_route` | Rush_hour × route | Continuous |

#### Graph Features (10 - for GNN only)
| Feature | Description | Type |
|---------|-------------|------|
| `node_lat` | Stop latitude | Continuous |
| `node_lon` | Stop longitude | Continuous |
| `node_dist_to_target` | Distance to target | Continuous |
| `node_avg_speed` | Average speed at stop | Continuous |
| `node_avg_dwell_time` | Average dwell time | Continuous |
| `node_visit_count` | Visit frequency | Continuous |
| `edge_distance` | Stop-to-stop distance | Continuous |
| `edge_avg_travel_time` | Historical travel time | Continuous |
| `edge_traffic_factor` | Congestion indicator | Continuous |
| `graph_density` | Route complexity | Continuous |

---

### Graph Construction (for GNN)

**Nodes**: 30 nodes per graph
- 29 official bus stops
- 1 current position node

**Node Features**: 6 features per node
- Latitude, Longitude
- Distance to target stop
- Average speed at stop
- Average dwell time
- Visit frequency

**Edges**: Sequential connections
- Stop_i → Stop_{i+1} (route sequence)
- Bidirectional for return routes
- Edge weights: historical travel time

**Edge Features**: 3 features per edge
- Distance between stops
- Average travel time
- Traffic congestion factor

**Graph-level Features**: 2 global features
- Total route distance
- Graph density (edges/nodes)

---

### Feature Importance (from XGBoost)

Top 10 most predictive features:

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `dist_to_stop_m` | 0.342 | Spatial |
| 2 | `time_to_stop_naive` | 0.187 | Derived |
| 3 | `speed_x_hour` | 0.124 | Interaction |
| 4 | `is_rush_hour` | 0.089 | Temporal |
| 5 | `avg_speed_last_10` | 0.067 | Speed |
| 6 | `stops_remaining` | 0.054 | Route |
| 7 | `speed_variance` | 0.041 | Speed |
| 8 | `route_progress` | 0.032 | Route |
| 9 | `stationary_ratio` | 0.028 | Speed |
| 10 | `hour` | 0.021 | Temporal |

**Key Insight**: Distance features alone account for 34% of predictive power, but ensemble of all features achieves 10% better accuracy than distance alone.

---

### Real-Time Preprocessing

For production API, preprocessing is optimized for single GPS points:

**Input**: Single GPS point
```python
{
    'latitude': 13.0123,
    'longitude': 80.2345,
    'speed': 25,
    'timestamp': '2026-01-09T10:30:00'
}
```

**Output**: 53-feature vector ready for prediction

**Latency**: <0.1ms (feature extraction only, excluding model inference)

**Key Optimizations:**
- No sliding window needed (stateless)
- Precomputed stop distances
- Cached route information
- Vectorized distance calculations

