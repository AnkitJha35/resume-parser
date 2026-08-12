from __future__ import annotations

from minio import Minio
from minio.error import S3Error

from app.core.config import Settings
from app.core.exceptions import StorageClientError


class MinioClient:
    def __init__(self, settings: Settings) -> None:
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self._bucket_name = settings.minio_bucket_name

    def download_pdf(self, storage_key: str) -> bytes:
        try:
            response = self._client.get_object(self._bucket_name, storage_key)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as exc:
            raise StorageClientError("Failed to download PDF from MinIO") from exc
