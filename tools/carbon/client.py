"""Carbon Arc REST client: Bearer auth, pagination, job polling."""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://api.carbonarc.co"
KEY_ENV_VARS = ("CARBON_API_KEY", "CARBONARC_API_KEY")


class CarbonError(RuntimeError):
    """Unusable response from the Carbon Arc API."""

    def __init__(self, status: int, path: str, detail: str):
        super().__init__(f"HTTP {status} on {path}: {detail}")
        self.status = status
        self.path = path
        self.detail = detail


def load_key() -> str:
    """API key from the environment, otherwise from a .env up the tree."""
    for name in KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    for folder in (Path.cwd(), *Path.cwd().parents):
        env = folder / ".env"
        if not env.is_file():
            continue
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("CARBON_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise CarbonError(0, "credentials", "set CARBON_API_KEY or add it to .env")


class CarbonClient:
    """JSON client for api.carbonarc.co; never logs the key."""

    def __init__(
        self,
        key: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = 60.0,
    ):
        self.key = key or load_key()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str, params: dict | None = None) -> object:
        """GET a path and return the parsed JSON body."""
        return self._request("GET", path, params=params)

    def post(self, path: str, body: object = None, params: dict | None = None) -> object:
        """POST a JSON body and return the parsed response."""
        return self._request("POST", path, params=params, body=body)

    def paged(self, path: str, params: dict | None = None, items_key: str = "items"):
        """Yield every item across the pages of a paginated listing."""
        params = dict(params or {})
        page = int(params.pop("page", 1))
        size = int(params.pop("size", 100))
        while True:
            payload = self.get(path, {**params, "page": page, "size": size})
            yield from payload.get(items_key, [])
            if page >= int(payload.get("total_pages", 1)):
                return
            page += 1

    def slow(
        self,
        path: str,
        params: dict | None = None,
        body: object = None,
        wait: float = 2.0,
        timeout: float = 900.0,
    ) -> object:
        """Run a query through the poll/job_id contract used by framework routes."""
        params = dict(params or {})
        response = self._request(
            "POST" if body is not None else "GET",
            path,
            params={**params, "poll": "true"},
            body=body,
        )
        job_id = response.get("job_id") if isinstance(response, dict) else None
        if not job_id:
            return response
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get(f"/v2/query-jobs/{job_id}")
            state = job.get("state")
            if state == "done":
                return self._request(
                    "POST" if body is not None else "GET",
                    path,
                    params={**params, "job_id": job_id},
                    body=body,
                )
            if state in ("failed", "cancelled"):
                raise CarbonError(0, path, f"job {job_id} {state}")
            time.sleep(wait)
        raise CarbonError(0, path, f"job {job_id} timed out after {timeout}s")

    def _request(
        self, method: str, path: str, params: dict | None = None, body: object = None
    ) -> object:
        url = self.base_url + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url += "?" + urllib.parse.urlencode(clean)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                return json.loads(reply.read() or b"null")
        except urllib.error.HTTPError as exc:
            raise CarbonError(
                exc.code, path, exc.read().decode("utf-8", "replace")[:300]
            ) from exc
