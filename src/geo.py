import math

def haversine_km(lat1, lon1, lat2, lon2):
    try:
        lat1=float(lat1); lon1=float(lon1); lat2=float(lat2); lon2=float(lon2)
    except Exception:
        return float("nan")
    R=6371.0
    dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c=2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c
