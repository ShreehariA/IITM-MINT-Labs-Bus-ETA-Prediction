"""
Stop Metadata - IIT Madras Campus
Complete stop information with coordinates and route mappings
"""

STOPS = {
    1: {
        'name': 'Velachery Gate IIT Campus Stop',
        'latitude': 12.988763557585395,
        'longitude': 80.22367960515129,
        'routes': ['VG_MAIN', 'VG_HOSTEL', 'MAIN_VG', 'HOSTEL_VG', 'HOSTEL_MAIN', 'MAIN_HOSTEL']
    },
    2: {
        'name': 'NAC2 Stop Towards GC',
        'latitude': 12.990236,
        'longitude': 80.227548,
        'routes': ['VG_MAIN', 'VG_HOSTEL', 'HOSTEL_MAIN', 'MAIN_HOSTEL', 'ED_RP']
    },
    3: {
        'name': 'Nac2 Towards Velachery Stop',
        'latitude': 12.9898636,
        'longitude': 80.2271109,
        'routes': ['MAIN_VG', 'HOSTEL_VG', 'HOSTEL_MAIN', 'MAIN_HOSTEL', 'RP_ED']
    },
    4: {
        'name': 'CRC Towards GC Stop',
        'latitude': 12.99089,
        'longitude': 80.230274,
        'routes': ['VG_MAIN', 'VG_HOSTEL', 'HOSTEL_MAIN', 'MAIN_HOSTEL', 'ED_RP']
    },
    5: {
        'name': 'CRC Towards Velachery Stop',
        'latitude': 12.9907968,
        'longitude': 80.2301666,
        'routes': ['MAIN_VG', 'HOSTEL_VG', 'HOSTEL_MAIN', 'MAIN_HOSTEL', 'RP_ED']
    },
    6: {
        'name': 'HSB Stop Towards Velachery Stop',
        'latitude': 12.9907678,
        'longitude': 80.2316895,
        'routes': ['MAIN_VG', 'HOSTEL_VG', 'HOSTEL_MAIN', 'MAIN_HOSTEL', 'RP_ED']
    },
    7: {
        'name': 'HSB Stop Towards GC',
        'latitude': 12.991037,
        'longitude': 80.232065,
        'routes': ['VG_MAIN', 'VG_HOSTEL', 'HOSTEL_MAIN', 'MAIN_HOSTEL', 'ED_RP']
    },
    8: {
        'name': 'KV Stop Towards Main Gate',
        'latitude': 12.991868573807471,
        'longitude': 80.23367454352295,
        'routes': ['VG_MAIN', 'HOSTEL_MAIN']
    },
    9: {
        'name': 'KV Stop Towards GC',
        'latitude': 12.99191,
        'longitude': 80.233785,
        'routes': ['MAIN_HOSTEL']
    },
    10: {
        'name': 'Postoffice Stop Towards Main Gate',
        'latitude': 12.9938868,
        'longitude': 80.2342876,
        'routes': ['VG_MAIN', 'HOSTEL_MAIN']
    },
    11: {
        'name': 'Postoffice Stop Towards GC',
        'latitude': 12.9939334,
        'longitude': 80.2345935,
        'routes': ['MAIN_HOSTEL']
    },
    12: {
        'name': 'E1 Stop Towards Main Gate',
        'latitude': 12.9960173,
        'longitude': 80.2359173,
        'routes': ['VG_MAIN', 'HOSTEL_MAIN']
    },
    13: {
        'name': 'E1 Stop Towards GC',
        'latitude': 12.996117,
        'longitude': 80.236184,
        'routes': ['MAIN_HOSTEL']
    },
    14: {
        'name': 'Vana Vani Stop Towards Main Gate',
        'latitude': 12.998704,
        'longitude': 80.2391799,
        'routes': ['VG_MAIN', 'HOSTEL_MAIN']
    },
    15: {
        'name': 'Vana Vani Stop Towards GC',
        'latitude': 12.999084,
        'longitude': 80.23938,
        'routes': ['MAIN_HOSTEL']
    },
    16: {
        'name': 'D1 Stop Towards Main Gate',
        'latitude': 13.002546,
        'longitude': 80.240091,
        'routes': ['VG_MAIN', 'HOSTEL_MAIN']
    },
    17: {
        'name': 'D1 Stop Towards GC',
        'latitude': 13.002679,
        'longitude': 80.240219,
        'routes': ['MAIN_HOSTEL']
    },
    18: {
        'name': 'Main Gate',
        'latitude': 13.00612964850378,
        'longitude': 80.24191299117685,
        'routes': ['VG_MAIN', 'HOSTEL_MAIN', 'MAIN_HOSTEL', 'MAIN_VG']
    },
    19: {
        'name': 'GC Stop Towards Hostel',
        'latitude': 12.991342,
        'longitude': 80.233637,
        'routes': ['VG_HOSTEL', 'MAIN_HOSTEL', 'ED_RP']
    },
    20: {
        'name': 'Library Stop',
        'latitude': 12.9907517,
        'longitude': 80.2334554,
        'routes': ['HOSTEL_MAIN', 'RP_ED']
    },
    21: {
        'name': 'OAT Stop Towards Hostel',
        'latitude': 12.989257,
        'longitude': 80.233031,
        'routes': ['MAIN_HOSTEL', 'RP_ED', 'ED_RP']
    },
    22: {
        'name': 'Gymkhana Stop Towards GC',
        'latitude': 12.9866372,
        'longitude': 80.2332951,
        'routes': ['HOSTEL_MAIN', 'RP_ED']
    },
    23: {
        'name': 'Gymkhana Stop Towards Hostel',
        'latitude': 12.986615,
        'longitude': 80.233366,
        'routes': ['VG_HOSTEL', 'MAIN_HOSTEL', 'ED_RP']
    },
    24: {
        'name': 'Narmada Stop Towards GC',
        'latitude': 12.9862759,
        'longitude': 80.2350107,
        'routes': ['HOSTEL_MAIN', 'HOSTEL_VG']
    },
    25: {
        'name': 'Narmada Stop Towards Hostel',
        'latitude': 12.986546,
        'longitude': 80.235301,
        'routes': ['VG_HOSTEL', 'MAIN_HOSTEL']
    },
    26: {
        'name': 'Jamuna and Ganga Stand',
        'latitude': 12.986566332825637,
        'longitude': 80.23855704439099,
        'routes': ['HOSTEL_MAIN', 'HOSTEL_VG', 'VG_HOSTEL', 'MAIN_HOSTEL']
    },
    28: {
        'name': 'RP Stand',
        'latitude': 12.990088,
        'longitude': 80.241799,
        'routes': ['RP_ED', 'ED_RP']
    },
    29: {
        'name': 'ED Stand',
        'latitude': 12.989874633072303,
        'longitude': 80.22646042038593,
        'routes': ['RP_ED', 'ED_RP']
    }
}

# Route sequences (ordered list of stop IDs)
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

def get_stop_info(stop_id: int) -> dict:
    """Get stop information by ID"""
    return STOPS.get(stop_id)

def get_route_stops(route_code: str) -> list:
    """Get ordered list of stop IDs for a route"""
    route = ROUTES.get(route_code)
    return route['stops'] if route else []

def get_upcoming_stops(current_position: tuple, route_code: str) -> list:
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
        dlat = math.radians(stop['latitude'] - lat)
        dlon = math.radians(stop['longitude'] - lon)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat)) * math.cos(math.radians(stop['latitude'])) * 
             math.sin(dlon/2)**2)
        distance = 2 * math.asin(math.sqrt(a)) * 6371000  # Earth radius in meters
        
        if distance < min_distance:
            min_distance = distance
            closest_idx = idx
    
    # Return stops from closest onwards
    return route_stops[closest_idx:]

if __name__ == "__main__":
    # Test
    print(f"Total stops: {len(STOPS)}")
    print(f"Total routes: {len(ROUTES)}")
    
    # Example: Get stops for HOSTEL_MAIN route
    stops = get_route_stops('HOSTEL_MAIN')
    print(f"\nHOSTEL_MAIN route has {len(stops)} stops:")
    for stop_id in stops:
        print(f"  {stop_id}: {STOPS[stop_id]['name']}")
