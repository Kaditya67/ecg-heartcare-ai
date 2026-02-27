import json
import logging

import redis
from django.conf import settings

logger = logging.getLogger(__name__)

# Centralized Redis client
redis_client = redis.StrictRedis(
    host=getattr(settings, "REDIS_HOST", "localhost"),
    port=getattr(settings, "REDIS_PORT", 6379),
    db=0,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
)


def set_ecg_wave(record_id, ecg_wave, timeout=120):
    """
    Store ECG wave in Redis with a timeout (default 2 mins).
    Silently fails if Redis is unavailable — caller gets DB fallback.
    """
    try:
        cache_key = f"ecg_wave:{record_id}"
        redis_client.set(cache_key, json.dumps(ecg_wave), ex=timeout)
    except Exception as e:
        logger.warning("Redis set failed for record %s: %s", record_id, e)


def get_ecg_wave(record_id):
    """
    Retrieve ECG wave from Redis if available.
    Returns None on any Redis error — caller falls back to DB.
    """
    try:
        cache_key = f"ecg_wave:{record_id}"
        data = redis_client.get(cache_key)
        return json.loads(data) if data else None
    except Exception as e:
        logger.warning("Redis get failed for record %s: %s", record_id, e)
        return None


def preload_page_waves(records, timeout=120):
    """
    Preload ECG waves for a list of records (e.g., 2 pages).
    """
    for record in records:
        if not get_ecg_wave(record.id):
            set_ecg_wave(record.id, record.ecg_wave, timeout=timeout)
