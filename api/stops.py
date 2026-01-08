"""
Stop Metadata and Route Information
Complete IIT Madras bus stop data with coordinates and route mappings
"""

STOPS = {
    1: {'name': 'Velachery Gate IIT Campus Stop', 'lat': 12.988763557585395, 'lon': 80.22367960515129},
    2: {'name': 'NAC2 Stop Towards GC', 'lat': 12.990236, 'lon': 80.227548},
    3: {'name': 'Nac2 Towards Velachery Stop', 'lat': 12.9898636, 'lon': 80.2271109},
    4: {'name': 'CRC Towards GC Stop', 'lat': 12.99089, 'lon': 80.230274},
    5: {'name': 'CRC Towards Velachery Stop', 'lat': 12.9907968, 'lon': 80.2301666},
    6: {'name': 'HSB Stop Towards Velachery Stop', 'lat': 12.9907678, 'lon': 80.2316895},
    7: {'name': 'HSB Stop Towards GC', 'lat': 12.991037, 'lon': 80.232065},
    8: {'name': 'KV Stop Towards Main Gate', 'lat': 12.991868573807471, 'lon': 80.23367454352295},
    9: {'name': 'KV Stop Towards GC', 'lat': 12.99191, 'lon': 80.233785},
    10: {'name': 'Postoffice Stop Towards Main Gate', 'lat': 12.9938868, 'lon': 80.2342876},
    11: {'name': 'Postoffice Stop Towards GC', 'lat': 12.9939334, 'lon': 80.2345935},
    12: {'name': 'E1 Stop Towards Main Gate', 'lat': 12.9960173, 'lon': 80.2359173},
    13: {'name': 'E1 Stop Towards GC', 'lat': 12.996117, 'lon': 80.236184},
    14: {'name': 'Vana Vani Stop Towards Main Gate', 'lat': 12.998704, 'lon': 80.2391799},
    15: {'name': 'Vana Vani Stop Towards GC', 'lat': 12.999084, 'lon': 80.23938},
    16: {'name': 'D1 Stop Towards Main Gate', 'lat': 13.002546, 'lon': 80.240091},
    17: {'name': 'D1 Stop Towards GC', 'lat': 13.002679, 'lon': 80.240219},
    18: {'name': 'Main Gate', 'lat': 13.00612964850378, 'lon': 80.24191299117685},
    19: {'name': 'GC Stop Towards Hostel', 'lat': 12.991342, 'lon': 80.233637},
    20: {'name': 'Library Stop', 'lat': 12.9907517, 'lon': 80.2334554},
    21: {'name': 'OAT Stop Towards Hostel', 'lat': 12.989257, 'lon': 80.233031},
    22: {'name': 'Gymkhana Stop Towards GC', 'lat': 12.9866372, 'lon': 80.2332951},
    23: {'name': 'Gymkhana Stop Towards Hostel', 'lat': 12.986615, 'lon': 80.233366},
    24: {'name': 'Narmada Stop Towards GC', 'lat': 12.9862759, 'lon': 80.2350107},
    25: {'name': 'Narmada Stop Towards Hostel', 'lat': 12.986546, 'lon': 80.235301},
    26: {'name': 'Jamuna and Ganga Stand', 'lat': 12.986566332825637, 'lon': 80.23855704439099},
    28: {'name': 'RP Stand', 'lat': 12.990088, 'lon': 80.241799},
    29: {'name': 'ED Stand', 'lat': 12.989874633072303, 'lon': 80.22646042038593}
}

ROUTES = {
    'HOSTEL_MAIN': {
        'label': 'Hostel → Main Gate',
        'stops': [26, 24, 22, 20, 6, 5, 3, 1, 2, 4, 7, 8, 10, 12, 14, 16, 18]
    },
    'MAIN_HOSTEL': {
        'label': 'Main Gate → Hostel',
        'stops': [18, 17, 15, 13, 11, 9, 6, 5, 3, 1, 2, 4, 7, 19, 21, 23, 25, 26]
    },
    'VG_MAIN': {
        'label': 'Velachery Gate → Main Gate',
        'stops': [1, 2, 4, 7, 8, 10, 12, 14, 16, 18]
    },
    'VG_HOSTEL': {
        'label': 'Velachery Gate → Hostel',
        'stops': [1, 2, 4, 7, 19, 21, 23, 25, 26]
    },
    'HOSTEL_VG': {
        'label': 'Hostel → Velachery Gate',
        'stops': [26, 24, 22, 20, 6, 5, 3, 1]
    },
    'MAIN_VG': {
        'label': 'Main Gate → Velachery Gate',
        'stops': [18, 17, 15, 13, 11, 9, 6, 5, 3, 1]
    },
    'RP_ED': {
        'label': 'Research Park → ED',
        'stops': [28, 22, 21, 20, 6, 5, 3, 29]
    },
    'ED_RP': {
        'label': 'ED → Research Park',
        'stops': [29, 2, 4, 7, 19, 21, 23, 28]
    }
}

def get_stop_info(stop_id):
    """Get stop information by ID"""
    return STOPS.get(stop_id)

def get_route_stops(route_code):
    """Get ordered list of stop IDs for a route"""
    route = ROUTES.get(route_code)
    return route['stops'] if route else []

def get_upcoming_stops(current_position, route_code):
    """
    Get upcoming stops based on current position and route.
    
    Args:
        current_position: (latitude, longitude)
        route_code: Route code (e.g., 'HOSTEL_MAIN')
        
    Returns:
        List of upcoming stop IDs
    """
    import math
    
    route_stops = get_route_stops(route_code)
    if not route_stops:
        return []
    
    # Find closest stop on route
    lat, lon = current_position
    min_distance = float('inf')
    closest_idx = 0
    
    for idx, stop_id in enumerate(route_stops):
        stop = STOPS[stop_id]
        # Haversine distance
        dlat = math.radians(stop['lat'] - lat)
        dlon = math.radians(stop['lon'] - lon)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat)) * math.cos(math.radians(stop['lat'])) * 
             math.sin(dlon/2)**2)
        distance = 2 * math.asin(math.sqrt(a)) * 6371000  # Earth radius in meters
        
        if distance < min_distance:
            min_distance = distance
            closest_idx = idx
    
    # Return stops from closest onwards
    return route_stops[closest_idx:]
