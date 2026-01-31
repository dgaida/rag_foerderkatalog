#!/usr/bin/env python3
"""
download_data.py - Lädt CSV und Index beim Render-Start herunter

Dieses Script wird vor main.py ausgeführt und stellt sicher,
dass alle benötigten Daten verfügbar sind.
"""

import sys
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, destination: Path, description: str = ""):
    """Lädt eine Datei mit Progress-Bar herunter."""

    if destination.exists():
        print(f"✅ {description} existiert bereits: {destination}")
        return True

    print(f"📥 Lade {description} herunter...")
    print(f"   URL: {url}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=description) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"✅ {description} erfolgreich heruntergeladen")
        return True

    except Exception as e:
        print(f"❌ Fehler beim Download von {description}: {e}")
        return False


def main():
    """Lädt alle benötigten Dateien herunter."""

    print("=" * 60)
    print("  RAG Förderkatalog - Daten-Download für Render")
    print("=" * 60)
    print()

    # GitHub Release URLs (anpassen an deine tatsächlichen Releases!)
    RELEASE_VERSION = "v0.3.0"  # Oder v0.3.1
    BASE_URL = f"https://github.com/dgaida/rag_foerderkatalog/releases/download/{RELEASE_VERSION}"

    downloads = [
        {
            "url": f"{BASE_URL}/rag_foerderkatalog_complete_v0.3.0.zip",
            "destination": Path("data/complete_release.zip"),
            "description": "Complete Release Package",
            "extract": True,
        }
    ]

    success = True

    for item in downloads:
        if not download_file(item["url"], item["destination"], item["description"]):
            success = False
            break

        # Extrahiere ZIP wenn nötig
        if item.get("extract") and item["destination"].suffix == ".zip":
            print(f"📦 Entpacke {item['description']}...")
            try:
                import zipfile

                with zipfile.ZipFile(item["destination"], "r") as zip_ref:
                    zip_ref.extractall(item["destination"].parent)
                print("✅ Erfolgreich entpackt")
            except Exception as e:
                print(f"❌ Fehler beim Entpacken: {e}")
                success = False

    print()
    print("=" * 60)

    if success:
        print("✅ Alle Daten erfolgreich heruntergeladen!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ Fehler beim Download der Daten")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
