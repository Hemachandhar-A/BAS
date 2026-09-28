#!/usr/bin/env python3
"""python scripts/gen_cert.py -- self-signed TLS cert for server/app.py
(IMPLEMENTATION_PLAN.md 5.9; essential-features.md #10 point 3). Uses the
``cryptography`` package so it works on every OS with nothing else
installed; browsers will warn once about the self-signed cert (expected).

Writes ``certs/cert.pem`` and ``certs/key.pem`` by default (already
git-ignored: ``.gitignore``'s ``*.pem``/``*.key`` patterns). Point
``RuntimeConfig.tls_cert``/``tls_key`` (``config/runtime.yaml``) at the
written paths to enable TLS; without both set, the server binds
``127.0.0.1`` regardless of ``host`` (AGENTS.md rule 14).
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def generate_cert(common_name: str, days_valid: int) -> tuple[bytes, bytes]:
    """Returns ``(cert_pem, key_pem)`` bytes for a self-signed cert with
    ``common_name`` as both subject and issuer, valid for ``days_valid``
    days from now, covering ``localhost``/``127.0.0.1`` as subject
    alternative names."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days_valid))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cert-path", default="certs/cert.pem", type=Path)
    parser.add_argument("--key-path", default="certs/key.pem", type=Path)
    parser.add_argument("--common-name", default="localhost")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing cert/key files"
    )
    args = parser.parse_args(argv)

    if not args.force and (args.cert_path.exists() or args.key_path.exists()):
        print(
            f"{args.cert_path} or {args.key_path} already exists; pass --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    cert_pem, key_pem = generate_cert(args.common_name, args.days)

    args.cert_path.parent.mkdir(parents=True, exist_ok=True)
    args.key_path.parent.mkdir(parents=True, exist_ok=True)
    args.cert_path.write_bytes(cert_pem)
    args.key_path.write_bytes(key_pem)

    print(f"Wrote {args.cert_path} and {args.key_path} (valid {args.days} days).")
    print(
        "Set tls_cert/tls_key to these paths in config/runtime.yaml to enable TLS; "
        "browsers will warn once about the self-signed cert."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
