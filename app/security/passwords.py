from passlib.context import CryptContext

_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return _context.verify(password, hashed)
    except ValueError:
        return False
