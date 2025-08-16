import redis
import json
from django.conf import settings

# Centralized Redis client
redis_client = redis.StrictRedis(
    host=getattr(settings, "REDIS_HOST", "localhost"),
    port=getattr(settings, "REDIS_PORT", 6379),
    db=0,
    decode_responses=True
)

def set_ecg_wave(record_id, ecg_wave, timeout=120):
    """
    Store ECG wave in Redis with a timeout (default 2 mins).
    """
    cache_key = f"ecg_wave:{record_id}"
    redis_client.set(cache_key, json.dumps(ecg_wave), ex=timeout)

def get_ecg_wave(record_id):
    """
    Retrieve ECG wave from Redis if available.
    """
    cache_key = f"ecg_wave:{record_id}"
    data = redis_client.get(cache_key)
    return json.loads(data) if data else None

def preload_page_waves(records, timeout=120):
    """
    Preload ECG waves for a list of records (e.g., 2 pages).
    """
    for record in records:
        if not get_ecg_wave(record.id):
            set_ecg_wave(record.id, record.ecg_wave, timeout=timeout)
