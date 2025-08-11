# src/reddit_sentiment/cache.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
import pandas as pd

try:
    import gcsfs  # noqa: F401
    _HAS_GCSFS = True
except Exception:
    _HAS_GCSFS = False


@dataclass(frozen=True)
class CacheKey:
    keyword: str
    days_back: int
    max_posts: int
    version: str = "v1"

    def as_str(self) -> str:
        safe = f"{self.keyword}:{self.days_back}:{self.max_posts}:{self.version}"
        return safe.replace(" ", "_").replace(":", "_")


class Cache(ABC):
    @abstractmethod
    def exists(self, key: CacheKey) -> bool: ...
    @abstractmethod
    def is_fresh(self, key: CacheKey, max_age_hours: int) -> bool: ...
    @abstractmethod
    def load(self, key: CacheKey) -> tuple[pd.DataFrame, pd.DataFrame]: ...
    @abstractmethod
    def save(self, key: CacheKey, posts: pd.DataFrame, comments: pd.DataFrame) -> None: ...


class LocalParquetCache(Cache):
    def __init__(self, base: Path):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: CacheKey):
        s = key.as_str()
        return self.base / f"{s}_posts.parquet", self.base / f"{s}_comments.parquet"

    def exists(self, key: CacheKey) -> bool:
        p, c = self._paths(key)
        return p.exists() and c.exists()

    def is_fresh(self, key: CacheKey, max_age_hours: int) -> bool:
        if max_age_hours <= 0:
            return True  # treat as always fresh if no TTL
        p, c = self._paths(key)
        if not (p.exists() and c.exists()):
            return False
        cutoff = time.time() - max_age_hours * 3600
        return p.stat().st_mtime >= cutoff and c.stat().st_mtime >= cutoff

    def load(self, key: CacheKey):
        p, c = self._paths(key)
        return pd.read_parquet(p), pd.read_parquet(c)

    def save(self, key: CacheKey, posts: pd.DataFrame, comments: pd.DataFrame):
        p, c = self._paths(key)
        posts.to_parquet(p, index=False)
        comments.to_parquet(c, index=False)


class GcsParquetCache(Cache):
    def __init__(self, bucket_prefix: str, project: str | None = None):
        if not _HAS_GCSFS:
            raise RuntimeError("gcsfs is required for GCS caching")
        import gcsfs  # local import to avoid hard dep in dev
        self.fs = gcsfs.GCSFileSystem(project=project)
        self.prefix = bucket_prefix.rstrip("/")

    def _paths(self, key: CacheKey):
        s = key.as_str()
        return f"{self.prefix}/{s}_posts.parquet", f"{self.prefix}/{s}_comments.parquet"

    def exists(self, key: CacheKey) -> bool:
        p, c = self._paths(key)
        return self.fs.exists(p) and self.fs.exists(c)

    def is_fresh(self, key: CacheKey, max_age_hours: int) -> bool:
        if max_age_hours <= 0:
            return True
        if not self.exists(key):
            return False
        p, c = self._paths(key)
        # Use newest of the two as the “age”
        info_p = self.fs.info(p)
        info_c = self.fs.info(c)
        newest = max(info_p.get("updated", 0), info_c.get("updated", 0))
        # gcsfs may return RFC3339 timestamps; fall back to exists() if parsing is messy
        try:
            import datetime as dt
            import dateutil.parser  # add python-dateutil to requirements if not already
            ts = dateutil.parser.isoparse(newest).timestamp()
            return (time.time() - ts) <= max_age_hours * 3600
        except Exception:
            return True  # conservatively treat as fresh if metadata parsing fails

    def load(self, key: CacheKey):
        import pandas as pd
        p, c = self._paths(key)
        return pd.read_parquet(p, filesystem=self.fs), pd.read_parquet(c, filesystem=self.fs)

    def save(self, key: CacheKey, posts: pd.DataFrame, comments: pd.DataFrame):
        p, c = self._paths(key)
        posts.to_parquet(p, index=False, filesystem=self.fs)
        comments.to_parquet(c, index=False, filesystem=self.fs)


def make_cache(backend: str, *, local_dir: Path, gcs_prefix: str | None, gcp_project: str | None):
    backend = (backend or "local").lower()
    if backend == "local":
        return LocalParquetCache(local_dir)
    if backend == "gcs":
        if not gcs_prefix:
            raise ValueError("GCS cache backend requires gcs_prefix")
        return GcsParquetCache(gcs_prefix, project=gcp_project)
    raise ValueError(f"Unknown cache backend: {backend}")
