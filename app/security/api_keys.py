import hashlib
import secrets

PREFIX = "cmt_"


def generate_api_key() -> str:
    return PREFIX + secrets.token_urlsafe(32)


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
