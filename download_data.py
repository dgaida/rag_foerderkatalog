#!/usr/bin/env python3
"""
download_data.py - Lädt CSV und Index beim Render-Start herunter

Dieses Script wird vor main.py ausgeführt und stellt sicher,
dass alle benötigten Daten verfügbar sind.
"""

import shutil
import sys
import zipfile
from pathlib import Path


def download_file_simple(url: str, destination: Path, description: str = ""):
    """Lädt eine Datei herunter (einfache Version ohne tqdm für Build)."""

    if destination.exists():
        print(f"✅ {description} existiert bereits: {destination}")
        return True

    print(f"📥 Lade {description} herunter...")
    print(f"   URL: {url}")

    try:
        import urllib.request

        destination.parent.mkdir(parents=True, exist_ok=True)

        # Download mit Progress
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r   Progress: {percent}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, destination, reporthook)
        print()  # Neue Zeile nach Progress

        print(f"✅ {description} erfolgreich heruntergeladen")
        return True

    except Exception as e:
        print(f"❌ Fehler beim Download von {description}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """Entpackt ein ZIP-Archiv."""
    print(f"📦 Entpacke {zip_path.name}...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Liste aller Dateien im ZIP
            file_list = zip_ref.namelist()
            print(f"   Gefunden: {len(file_list)} Dateien")

            # Entpacke alles
            zip_ref.extractall(extract_to)

        print(f"✅ Erfolgreich nach {extract_to} entpackt")

        # Zeige Struktur
        print("   Entpackte Struktur:")
        for item in extract_to.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"     - {item.relative_to(extract_to)} ({size_mb:.1f} MB)")

        return True

    except Exception as e:
        print(f"❌ Fehler beim Entpacken: {e}")
        return False


def move_files_from_zip_structure(base_path: Path):
    """
    Verschiebt Dateien aus möglicher Unterordner-Struktur an die richtige Stelle.

    ZIP könnte enthalten:
    - rag_foerderkatalog_complete_v0.3.0/
      - input/
        - foerderkatalog_export.csv
      - data/
        - vector_hf.index
        - embeddings_map_hf.json

    Oder direkt:
    - input/
    - data/
    """
    print("🔍 Prüfe und reorganisiere Dateistruktur...")

    # Suche nach input/ und data/ Ordnern
    input_dirs = list(base_path.rglob("input"))
    data_dirs = list(base_path.rglob("data"))

    # Ziel-Verzeichnisse
    target_input = base_path / "input"
    target_data = base_path / "data"

    # Verschiebe input/ wenn nötig
    for input_dir in input_dirs:
        if input_dir != target_input and input_dir.is_dir():
            print(f"   Verschiebe {input_dir} → {target_input}")
            target_input.mkdir(parents=True, exist_ok=True)

            for file in input_dir.iterdir():
                if file.is_file():
                    shutil.copy2(file, target_input / file.name)
                    print(f"     ✓ {file.name}")

    # Verschiebe data/ wenn nötig
    for data_dir in data_dirs:
        if data_dir != target_data and data_dir.is_dir():
            print(f"   Verschiebe {data_dir} → {target_data}")
            target_data.mkdir(parents=True, exist_ok=True)

            for file in data_dir.iterdir():
                if file.is_file():
                    shutil.copy2(file, target_data / file.name)
                    print(f"     ✓ {file.name}")

    print("✅ Dateistruktur bereinigt")


def verify_files():
    """Prüft, ob alle benötigten Dateien vorhanden sind."""
    print("\n🔍 Verifiziere Dateien...")

    base_path = Path("/opt/render/project/src")

    required_files = [
        base_path / "input" / "foerderkatalog_export.csv",
        base_path / "data" / "vector_hf.index",
        base_path / "data" / "embeddings_map_hf.json",
    ]

    all_present = True

    for file in required_files:
        if file.exists():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   ✅ {file.name} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ FEHLT: {file.name}")
            all_present = False

    return all_present


def main():
    """Lädt alle benötigten Dateien herunter."""

    print("=" * 60)
    print("  RAG Förderkatalog - Daten-Download für Render")
    print("=" * 60)
    print()

    # Arbeitsverzeichnis
    base_path = Path("/opt/render/project/src")
    print(f"📁 Arbeitsverzeichnis: {base_path}")
    print(f"   Existiert: {base_path.exists()}")
    print()

    # Prüfe ob Dateien bereits vorhanden
    if verify_files():
        print("\n✅ Alle Dateien bereits vorhanden - überspringe Download")
        sys.exit(0)

    # GitHub Release URLs
    # WICHTIG: Passe diese URLs an deine tatsächlichen GitHub Releases an!
    RELEASE_VERSION = "v0.3.1"
    GITHUB_USER = "dgaida"
    REPO_NAME = "rag_foerderkatalog"

    # Mögliche Download-URLs (passe an!)
    possible_urls = [
        # Direkte GitHub Release URL (bevorzugt)
        f"https://github.com/{GITHUB_USER}/{REPO_NAME}/releases/download/{RELEASE_VERSION}/rag_foerderkatalog_index_{RELEASE_VERSION}.zip",
        # Alternative: Cloudflare R2 / AWS S3 (wenn du das nutzt)
        # "https://pub-xxxxx.r2.dev/rag_foerderkatalog_complete_v0.3.0.zip",
    ]

    # Versuche Download
    zip_path = base_path / "downloads" / "data.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    download_success = False

    for url in possible_urls:
        print("\n🔄 Versuche Download von:")
        print(f"   {url}")

        if download_file_simple(url, zip_path, "Complete Data Package"):
            download_success = True
            break

    if not download_success:
        print("\n" + "=" * 60)
        print("❌ FEHLER: Download fehlgeschlagen!")
        print("=" * 60)
        print()
        print("MÖGLICHE LÖSUNGEN:")
        print()
        print("1. GitHub Release erstellen:")
        print("   - Gehe zu: https://github.com/dgaida/rag_foerderkatalog/releases")
        print(f"   - Erstelle neues Release mit Tag: {RELEASE_VERSION}")
        print("   - Lade ZIP hoch mit:")
        print("     - input/foerderkatalog_export.csv")
        print("     - data/vector_hf.index")
        print("     - data/embeddings_map_hf.json")
        print()
        print("2. Oder nutze Cloud Storage (Cloudflare R2, AWS S3)")
        print("   - Lade ZIP hoch")
        print("   - Passe URL in download_data.py an")
        print()
        print("3. Oder nutze Render Disk Storage:")
        print("   - Lade Dateien manuell in Render-Dashboard hoch")
        print()
        sys.exit(1)

    # Entpacke ZIP
    print()
    if not extract_zip(zip_path, base_path):
        print("❌ Entpacken fehlgeschlagen")
        sys.exit(1)

    # Reorganisiere Dateien falls nötig
    move_files_from_zip_structure(base_path)

    # Finale Verifikation
    if not verify_files():
        print("\n❌ FEHLER: Nicht alle Dateien konnten extrahiert werden")
        sys.exit(1)

    # Cleanup
    print("\n🧹 Räume auf...")
    shutil.rmtree(zip_path.parent, ignore_errors=True)

    print()
    print("=" * 60)
    print("✅ Alle Daten erfolgreich heruntergeladen und vorbereitet!")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
