# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from pathlib import Path
import sys
import requests
from urllib.parse import urlparse


def _download_error_message(filename, url, primary_error, backup_url=None, backup_error=None):
    details = [f"Failed to download {filename}."]

    if primary_error is not None:
        details.append(
            f"Primary URL failed ({url}): "
            f"{type(primary_error).__name__}: {primary_error}"
        )

    if backup_url and backup_error is not None:
        details.append(
            f"Backup URL failed ({backup_url}): "
            f"{type(backup_error).__name__}: {backup_error}"
        )

    cert_or_proxy_issue = any(
        isinstance(err, (requests.exceptions.ProxyError, requests.exceptions.SSLError))
        for err in (primary_error, backup_error)
        if err is not None
    )
    if not cert_or_proxy_issue:
        lowered = " ".join(
            str(err).lower() for err in (primary_error, backup_error) if err is not None
        )
        cert_or_proxy_issue = any(
            keyword in lowered for keyword in ("certificate", "ssl", "tls", "proxy")
        )

    if cert_or_proxy_issue:
        details.append(
            "This can happen on work or school machines where a VPN, proxy, or "
            "antivirus tool intercepts HTTPS certificates."
        )

    details.append(
        "See the troubleshooting guide: "
        "https://github.com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md "
        "(especially the 'File Download Issues' section)."
    )
    return "\n".join(details)


def download_file(url, out_dir=".", backup_url=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    dest = out_dir / filename

    def try_download(u):
        try:
            with requests.get(u, stream=True, timeout=30) as r:
                r.raise_for_status()
                size_remote = int(r.headers.get("Content-Length", 0))

                # Skip download if already complete
                if dest.exists() and size_remote and dest.stat().st_size == size_remote:
                    print(f"✓ {dest} already up-to-date")
                    return True, None

                # Download in 1 MiB chunks with progress display
                block = 1024 * 1024
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if size_remote:
                            pct = downloaded * 100 // size_remote
                            sys.stdout.write(
                                f"\r{filename}: {pct:3d}% "
                                f"({downloaded // (1024*1024)} MiB / "
                                f"{size_remote // (1024*1024)} MiB)"
                            )
                            sys.stdout.flush()
                if size_remote:
                    sys.stdout.write("\n")
            return True, None
        except requests.RequestException as exc:
            return False, exc

    # Try main URL first
    success, primary_error = try_download(url)
    if success:
        return dest

    # Try backup URL if provided
    backup_error = None
    if backup_url:
        print(f"Primary URL ({url}) failed.\nTrying backup URL ({backup_url})...")
        success, backup_error = try_download(backup_url)
        if success:
            return dest

    message = _download_error_message(
        filename=filename,
        url=url,
        primary_error=primary_error,
        backup_url=backup_url,
        backup_error=backup_error,
    )
    raise RuntimeError(message) from (backup_error or primary_error)
