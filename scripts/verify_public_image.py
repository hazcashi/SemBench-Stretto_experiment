"""Verify a pinned GHCR manifest using only anonymous registry access."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="ghcr.io/owner/image@sha256:...")
    args = parser.parse_args()
    reference = args.image or (Path(__file__).resolve().parents[1] / "docker/image.txt").read_text().strip()
    if not reference.startswith("ghcr.io/") or "@sha256:" not in reference:
        parser.error("An immutable ghcr.io/...@sha256:... reference is required")
    repository, digest = reference.removeprefix("ghcr.io/").split("@", 1)
    query = urlencode({"service": "ghcr.io", "scope": f"repository:{repository}:pull"})
    # Deliberately no GitHub credential manager, Docker config or env token.
    with urlopen(f"https://ghcr.io/token?{query}", timeout=30) as response:
        token = json.load(response)["token"]
    request = Request(f"https://ghcr.io/v2/{repository}/manifests/{digest}", headers={
        "Authorization": f"Bearer {token}",
        "Accept": ", ".join(("application/vnd.oci.image.index.v1+json",
                             "application/vnd.oci.image.manifest.v1+json",
                             "application/vnd.docker.distribution.manifest.list.v2+json",
                             "application/vnd.docker.distribution.manifest.v2+json")),
    })
    with urlopen(request, timeout=30) as response:
        content = response.read()
    actual = "sha256:" + hashlib.sha256(content).hexdigest()
    if actual != digest:
        raise RuntimeError(f"Manifest digest mismatch: expected {digest}, got {actual}")
    manifest = json.loads(content)
    platforms = [item.get("platform") for item in manifest.get("manifests", [])]
    if platforms and not any(p and p.get("os") == "linux" and p.get("architecture") == "amd64" for p in platforms):
        raise RuntimeError(f"No Linux/amd64 image: {platforms}")
    print(json.dumps({"image": reference, "anonymous_access": "passed",
                      "manifest_sha256": actual, "platforms": platforms}, indent=2))


if __name__ == "__main__":
    main()
