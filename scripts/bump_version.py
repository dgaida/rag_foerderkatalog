#!/usr/bin/env python3
"""
Version Bump Script für RAG Förderkatalog

Aktualisiert die Version in allen relevanten Dateien:
- src/version.py
- pyproject.toml
- CHANGELOG.md

Usage:
    python scripts/bump_version.py patch  # 0.1.0 -> 0.1.1
    python scripts/bump_version.py minor  # 0.1.0 -> 0.2.0
    python scripts/bump_version.py major  # 0.1.0 -> 1.0.0
    python scripts/bump_version.py 1.2.3  # Direkte Version
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parsed eine Version-String in (major, minor, patch)."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str)
    if not match:
        raise ValueError(f"Ungültiges Versionsformat: {version_str}")
    return tuple(map(int, match.groups()))


def format_version(major: int, minor: int, patch: int) -> str:
    """Formatiert eine Version als String."""
    return f"{major}.{minor}.{patch}"


def bump_version(current: str, bump_type: str) -> str:
    """Erhöht die Version basierend auf bump_type."""
    major, minor, patch = parse_version(current)

    if bump_type == "major":
        return format_version(major + 1, 0, 0)
    elif bump_type == "minor":
        return format_version(major, minor + 1, 0)
    elif bump_type == "patch":
        return format_version(major, minor, patch + 1)
    else:
        # Direktes Setzen einer Version
        try:
            parse_version(bump_type)  # Validierung
            return bump_type
        except ValueError:
            raise ValueError(f"Ungültiger bump_type: {bump_type}")


def get_current_version() -> str:
    """Liest die aktuelle Version aus src/version.py."""
    version_file = Path("src/version.py")
    if not version_file.exists():
        raise FileNotFoundError("src/version.py nicht gefunden!")

    content = version_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Version nicht in src/version.py gefunden!")

    return match.group(1)


def update_version_py(new_version: str) -> None:
    """Aktualisiert src/version.py."""
    version_file = Path("src/version.py")
    content = version_file.read_text()

    # Update __version__
    content = re.sub(r'__version__\s*=\s*["\'][^"\']+["\']', f'__version__ = "{new_version}"', content)

    # Update __version_info__
    major, minor, patch = parse_version(new_version)
    content = re.sub(r"__version_info__\s*=.*", f"__version_info__ = ({major}, {minor}, {patch})", content)

    version_file.write_text(content)
    print(f"✅ src/version.py aktualisiert: {new_version}")


def update_pyproject_toml(new_version: str) -> None:
    """Aktualisiert pyproject.toml."""
    pyproject_file = Path("pyproject.toml")
    content = pyproject_file.read_text()

    content = re.sub(r'version\s*=\s*["\'][^"\']+["\']', f'version = "{new_version}"', content, count=1)

    pyproject_file.write_text(content)
    print(f"✅ pyproject.toml aktualisiert: {new_version}")


def update_changelog(new_version: str, old_version: str) -> None:
    """Fügt einen neuen Eintrag im CHANGELOG.md hinzu."""
    changelog_file = Path("CHANGELOG.md")

    if not changelog_file.exists():
        print("⚠️  CHANGELOG.md nicht gefunden, wird erstellt")
        changelog_file.write_text("# Changelog\n\n")

    content = changelog_file.read_text()
    today = datetime.now().strftime("%Y-%m-%d")

    # Erstelle neuen Eintrag
    new_entry = f"""## [{new_version}] - {today}

### Changed
- Version Bump von {old_version} zu {new_version}

### Added
- TODO: Fügen Sie hier neue Features hinzu

### Fixed
- TODO: Fügen Sie hier Bugfixes hinzu

---

"""

    # Füge nach dem Header ein
    if "## [" in content:
        # Es gibt bereits Einträge
        parts = content.split("## [", 1)
        content = parts[0] + new_entry + "## [" + parts[1]
    else:
        # Erster Eintrag
        content += "\n" + new_entry

    changelog_file.write_text(content)
    print(f"✅ CHANGELOG.md aktualisiert mit neuer Version {new_version}")
    print("⚠️  Bitte ergänzen Sie die TODOs im CHANGELOG.md!")


def create_git_tag(version: str, push: bool = False) -> None:
    """Erstellt einen Git-Tag für die neue Version."""
    import subprocess

    tag_name = f"v{version}"
    message = f"Release version {version}"

    try:
        # Erstelle Tag
        subprocess.run(["git", "tag", "-a", tag_name, "-m", message], check=True, capture_output=True)
        print(f"✅ Git-Tag erstellt: {tag_name}")

        if push:
            subprocess.run(["git", "push", "origin", tag_name], check=True, capture_output=True)
            print(f"✅ Tag gepusht: {tag_name}")
        else:
            print(f"💡 Push mit: git push origin {tag_name}")

    except subprocess.CalledProcessError as e:
        print(f"⚠️  Git-Tag konnte nicht erstellt werden: {e}")
        print(f"   Erstellen Sie den Tag manuell: git tag -a {tag_name} -m '{message}'")


def main():
    """Hauptfunktion."""
    if len(sys.argv) < 2:
        print("Usage: python bump_version.py [major|minor|patch|X.Y.Z]")
        sys.exit(1)

    bump_type = sys.argv[1]
    create_tag = "--tag" in sys.argv
    push_tag = "--push" in sys.argv

    print("════════════════════════════════════════════════════════════")
    print("  RAG Förderkatalog - Version Bump")
    print("════════════════════════════════════════════════════════════")
    print()

    try:
        # Aktuelle Version auslesen
        current_version = get_current_version()
        print(f"📌 Aktuelle Version: {current_version}")

        # Neue Version berechnen
        new_version = bump_version(current_version, bump_type)
        print(f"🆕 Neue Version:     {new_version}")
        print()

        # Bestätigung
        response = input(f"Version von {current_version} auf {new_version} ändern? [y/N]: ")
        if response.lower() != "y":
            print("❌ Abgebrochen")
            sys.exit(0)

        print()
        print("🔄 Aktualisiere Dateien...")
        print()

        # Update Dateien
        update_version_py(new_version)
        update_pyproject_toml(new_version)
        update_changelog(new_version, current_version)

        print()
        print("════════════════════════════════════════════════════════════")
        print("✅ Version erfolgreich aktualisiert!")
        print("════════════════════════════════════════════════════════════")
        print()
        print("📋 Nächste Schritte:")
        print("   1. CHANGELOG.md bearbeiten und TODOs ausfüllen")
        print(f"   2. Änderungen committen: git commit -am 'Bump version to {new_version}'")
        print(f"   3. Tag erstellen: git tag -a v{new_version} -m 'Release v{new_version}'")
        print(f"   4. Pushen: git push && git push origin v{new_version}")
        print()

        # Optional: Git-Tag erstellen
        if create_tag:
            print("🏷️  Erstelle Git-Tag...")
            create_git_tag(new_version, push=push_tag)
            print()

    except Exception as e:
        print(f"❌ Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
