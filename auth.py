"""Supabase access-token verification.

Supabase can sign access tokens with the new **asymmetric** keys (ES256/RS256,
verified against the project JWKS) or the legacy **HS256** shared secret. We try,
in order:

  1. JWKS (asymmetric) — fetch + cache the project's public keys and verify.
  2. HS256 with ``SUPABASE_JWT_SECRET`` — the legacy shared secret.
  3. ``GET {SUPABASE_URL}/auth/v1/user`` with the bearer token + service-role
     ``apikey`` — ask Supabase to validate, read the user id from the response.

``verify_token`` returns the token claims (with a ``sub`` = user id) or raises
``AuthError``. No secrets are hardcoded — everything comes from the environment.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

import jwt
from jwt import PyJWKClient

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_ASYMMETRIC_ALGS = ("ES256", "RS256")

# Lazily-built JWKS client (fetches + caches the project's public keys). Exposed
# at module level so tests can inject a fake.
_jwk_client: Optional[PyJWKClient] = None


class AuthError(Exception):
    """Raised when a token cannot be verified by any method."""


def _get_jwk_client() -> Optional[PyJWKClient]:
    global _jwk_client
    if _jwk_client is None and SUPABASE_URL:
        _jwk_client = PyJWKClient(
            f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", cache_keys=True
        )
    return _jwk_client


def _verify_via_userinfo(token: str) -> Optional[dict]:
    """Last-resort validation: let Supabase check the token for us."""
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
        return None
    req = urllib.request.Request(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_SERVICE_ROLE_KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:  # noqa: BLE001 - any failure -> not verified
        return None
    uid = data.get("id")
    if uid:
        return {"sub": uid, "email": data.get("email")}
    return None


def verify_token(token: str) -> dict:
    """Verify a Supabase access token and return its claims, or raise AuthError."""
    if not token:
        raise AuthError("Empty token")
    try:
        header = jwt.get_unverified_header(token)
    except Exception as exc:  # noqa: BLE001
        raise AuthError("Malformed token") from exc
    alg = header.get("alg", "")

    # 1) Asymmetric (new Supabase signing keys) verified against the JWKS.
    if alg in _ASYMMETRIC_ALGS:
        client = _get_jwk_client()
        if client is not None:
            try:
                key = client.get_signing_key_from_jwt(token).key
                return jwt.decode(
                    token, key, algorithms=list(_ASYMMETRIC_ALGS),
                    options={"verify_aud": False},
                )
            except Exception:  # noqa: BLE001 - fall through to other methods
                pass

    # 2) Legacy HS256 shared secret.
    if alg == "HS256" and SUPABASE_JWT_SECRET:
        try:
            return jwt.decode(
                token, SUPABASE_JWT_SECRET, algorithms=["HS256"],
                options={"verify_aud": False},
            )
        except Exception:  # noqa: BLE001
            pass

    # 3) Ask Supabase directly.
    claims = _verify_via_userinfo(token)
    if claims is not None:
        return claims

    raise AuthError("Token could not be verified")


def user_id_from_token(token: str) -> str:
    """Verify a token and return its subject (user id), or raise AuthError."""
    claims = verify_token(token)
    sub = claims.get("sub")
    if not sub:
        raise AuthError("Token has no subject")
    return str(sub)


def delete_auth_user(user_id: str) -> bool:
    """Delete a Supabase auth user via the admin API (GDPR account deletion).

    Requires the service-role key (admin scope); returns True on success, False if
    the key is unset or Supabase rejects the request. The caller decides how to
    treat a False (the user's data is still removed regardless).
    """
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
        return False
    req = urllib.request.Request(
        f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
        },
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception:  # noqa: BLE001 - any failure -> report not-removed
        return False
