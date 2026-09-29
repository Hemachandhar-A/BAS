"""tests/unit/scripts/test_gen_cert.py -- scripts/gen_cert.py (F10;
IMPLEMENTATION_PLAN.md 5.9, essential-features.md #10 point 3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography import x509

from scripts.gen_cert import generate_cert, main


@pytest.mark.F10
def test_generate_cert_produces_a_loadable_self_signed_cert() -> None:
    cert_pem, key_pem = generate_cert("localhost", days_valid=30)
    cert = x509.load_pem_x509_certificate(cert_pem)
    assert cert.subject == cert.issuer
    assert cert.public_key() is not None
    assert b"PRIVATE KEY" in key_pem


@pytest.mark.F10
def test_main_writes_cert_and_key_files(tmp_path: Path) -> None:
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    rc = main(["--cert-path", str(cert_path), "--key-path", str(key_path)])
    assert rc == 0
    assert cert_path.exists()
    assert key_path.exists()
    x509.load_pem_x509_certificate(cert_path.read_bytes())


@pytest.mark.F10
def test_main_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    assert main(["--cert-path", str(cert_path), "--key-path", str(key_path)]) == 0
    rc = main(["--cert-path", str(cert_path), "--key-path", str(key_path)])
    assert rc == 1


@pytest.mark.F10
def test_main_overwrites_with_force(tmp_path: Path) -> None:
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    main(["--cert-path", str(cert_path), "--key-path", str(key_path)])
    rc = main(["--cert-path", str(cert_path), "--key-path", str(key_path), "--force"])
    assert rc == 0


@pytest.mark.F10
def test_main_rejects_zero_days(tmp_path: Path) -> None:
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    rc = main(["--cert-path", str(cert_path), "--key-path", str(key_path), "--days", "0"])
    assert rc == 1
    assert not cert_path.exists()
    assert not key_path.exists()


@pytest.mark.F10
def test_main_rejects_negative_days(tmp_path: Path) -> None:
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    rc = main(["--cert-path", str(cert_path), "--key-path", str(key_path), "--days", "-5"])
    assert rc == 1
    assert not cert_path.exists()


@pytest.mark.F10
def test_main_rejects_identical_cert_and_key_paths(tmp_path: Path) -> None:
    same_path = tmp_path / "both.pem"
    rc = main(["--cert-path", str(same_path), "--key-path", str(same_path)])
    assert rc == 1
    assert not same_path.exists()
