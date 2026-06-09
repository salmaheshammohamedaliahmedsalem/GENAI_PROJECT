"""
http_client.py — Robust HTTP GET with retry and exponential backoff.

Handles transient errors from academic APIs:
  - 429 Too Many Requests   → respects Retry-After header
  - 5xx Server Errors       → exponential backoff up to 16 s
  - Network timeouts        → retried up to `retries` times

Public API
----------
    http_get(url, params, headers, retries, timeout) -> requests.Response
"""
from __future__ import annotations

import time
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

from src.rag.mix.config import HTTP_TIMEOUT


def http_get(
    url: str,
    params=None,
    headers=None,
    retries: int = 4,
    timeout: int = HTTP_TIMEOUT,
):
    """
    GET *url* with automatic retry on transient HTTP errors.

    Returns a successful requests.Response (2xx).
    Raises the last exception after all retries are exhausted.
    """
    if requests is None:
        raise RuntimeError("requests is not installed — run: pip install requests")

    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After", "")
                delay = float(ra) if ra.replace(".", "", 1).isdigit() else min(2 ** attempt, 16)
                last_exc = requests.exceptions.HTTPError(
                    f"HTTP {r.status_code} from {urlparse(url).netloc}"
                )
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(min(2 ** attempt, 16))

    raise last_exc if last_exc else RuntimeError("All retries exhausted")
