"""
auth.py — JWT authentication + subscriber management
=====================================================
Users are stored in users.json (simple file DB — swap for Postgres/SQLite in production).

Roles:
  admin      → full access: can add/remove/suspend subscribers, switch data source,
                adjust thresholds, view all signals
  subscriber → read-only access to signals dashboard; must be approved by admin

Admin credentials are set via environment variables (or .env file):
  ADMIN_USERNAME   default: admin
  ADMIN_PASSWORD   default: changeme123   ← CHANGE THIS before deploy
  JWT_SECRET       default: supersecret   ← CHANGE THIS before deploy
"""

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY       = os.getenv("JWT_SECRET",       "supersecret-change-in-production")
ALGORITHM        = "HS256"
TOKEN_EXPIRE_MIN = 60 * 8   # 8 hours

ADMIN_USERNAME   = os.getenv("ADMIN_USERNAME",   "admin")
ADMIN_PASSWORD   = os.getenv("ADMIN_PASSWORD",   "changeme123")

USERS_FILE = Path(__file__).parent / "users.json"

pwd_ctx   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2    = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ── Pydantic models ────────────────────────────────────────────────────────────
class UserIn(BaseModel):
    username: str
    password: str
    email:    str = ""
    name:     str = ""

class UserOut(BaseModel):
    id:        str
    username:  str
    email:     str
    name:      str
    role:      str        # "admin" | "subscriber"
    status:    str        # "active" | "suspended" | "pending"
    created:   str

class Token(BaseModel):
    access_token: str
    token_type:   str
    role:         str
    username:     str
    name:         str

# ── User store (JSON file) ─────────────────────────────────────────────────────
def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    return json.loads(USERS_FILE.read_text())

def _save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))

def _ensure_admin():
    """Create the admin user on first boot if not present."""
    users = _load_users()
    if ADMIN_USERNAME not in users:
        users[ADMIN_USERNAME] = {
            "id":       str(uuid.uuid4()),
            "username": ADMIN_USERNAME,
            "hashed_pw": pwd_ctx.hash(ADMIN_PASSWORD),
            "email":    "",
            "name":     "Admin",
            "role":     "admin",
            "status":   "active",
            "created":  datetime.now(timezone.utc).isoformat(),
        }
        _save_users(users)

_ensure_admin()

# ── Auth helpers ───────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)

def create_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    expire  = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=TOKEN_EXPIRE_MIN))
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(username: str, password: str) -> Optional[dict]:
    users = _load_users()
    user  = users.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_pw"]):
        return None
    if user["status"] != "active":
        return None
    return user

async def get_current_user(token: str = Depends(oauth2)) -> dict:
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise cred_exc
    except JWTError:
        raise cred_exc
    users = _load_users()
    user  = users.get(username)
    if not user or user["status"] != "active":
        raise cred_exc
    return user

async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# ── Subscriber CRUD ────────────────────────────────────────────────────────────
def list_subscribers() -> list[UserOut]:
    users = _load_users()
    return [
        UserOut(**{k: v for k, v in u.items() if k != "hashed_pw"})
        for u in users.values()
        if u["role"] == "subscriber"
    ]

def add_subscriber(data: UserIn) -> UserOut:
    users = _load_users()
    if data.username in users:
        raise HTTPException(status_code=409, detail="Username already exists")
    uid = str(uuid.uuid4())
    record = {
        "id":        uid,
        "username":  data.username,
        "hashed_pw": hash_password(data.password),
        "email":     data.email,
        "name":      data.name or data.username,
        "role":      "subscriber",
        "status":    "active",   # admin creates → immediately active
        "created":   datetime.now(timezone.utc).isoformat(),
    }
    users[data.username] = record
    _save_users(users)
    return UserOut(**{k: v for k, v in record.items() if k != "hashed_pw"})

def update_subscriber_status(username: str, new_status: str) -> UserOut:
    if new_status not in ("active", "suspended"):
        raise HTTPException(status_code=400, detail="Status must be active or suspended")
    users = _load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if users[username]["role"] == "admin":
        raise HTTPException(status_code=403, detail="Cannot modify admin status")
    users[username]["status"] = new_status
    _save_users(users)
    return UserOut(**{k: v for k, v in users[username].items() if k != "hashed_pw"})

def delete_subscriber(username: str):
    users = _load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if users[username]["role"] == "admin":
        raise HTTPException(status_code=403, detail="Cannot delete admin")
    del users[username]
    _save_users(users)

def change_password(username: str, new_password: str):
    users = _load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    users[username]["hashed_pw"] = hash_password(new_password)
    _save_users(users)
