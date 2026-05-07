import logging
import time
import uuid


api_request_logger = logging.getLogger("api.requests")


class ApiRequestLoggingMiddleware:
    """
    Logs every /api request at the Django middleware layer so requests are
    visible even when a view raises before returning normally.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.path.startswith("/api/"):
            return self.get_response(request)

        request_id = uuid.uuid4().hex[:8]
        started_at = time.perf_counter()
        client_ip = request.META.get("HTTP_X_FORWARDED_FOR", request.META.get("REMOTE_ADDR", "-"))
        referer = request.META.get("HTTP_REFERER", "-")
        user_agent = request.META.get("HTTP_USER_AGENT", "-")
        user = getattr(request, "user", None)
        username = user.username if getattr(user, "is_authenticated", False) else "anonymous"

        api_request_logger.info(
            "[%s] -> %s %s ip=%s user=%s referer=%s ua=%s",
            request_id,
            request.method,
            request.path,
            client_ip,
            username,
            referer,
            user_agent,
        )

        try:
            response = self.get_response(request)
        except Exception:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            api_request_logger.exception(
                "[%s] !! %s %s status=500 duration_ms=%s",
                request_id,
                request.method,
                request.path,
                duration_ms,
            )
            raise

        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        api_request_logger.info(
            "[%s] <- %s %s status=%s duration_ms=%s",
            request_id,
            request.method,
            request.path,
            response.status_code,
            duration_ms,
        )
        return response
