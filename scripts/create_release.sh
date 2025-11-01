#!/bin/bash
# Release Script für RAG Förderkatalog v0.1.0
# Erstellt ZIP-Archiv mit FAISS-Index und Mapping-Datei

set -e  # Exit bei Fehler

VERSION="0.1.0"
RELEASE_NAME="rag_foerderkatalog_index_v${VERSION}"
DATA_DIR="data"
RELEASE_DIR="releases"
ZIP_FILE="${RELEASE_DIR}/${RELEASE_NAME}.zip"

echo "════════════════════════════════════════════════════════════"
echo "  RAG Förderkatalog - Release Creator v${VERSION}"
echo "════════════════════════════════════════════════════════════"
echo ""

# Prüfe ob data/ Verzeichnis existiert
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: $DATA_DIR Verzeichnis nicht gefunden!"
    exit 1
fi

# Prüfe ob vector.index existiert
if [ ! -f "$DATA_DIR/vector.index" ]; then
    echo "❌ Error: $DATA_DIR/vector.index nicht gefunden!"
    echo "   Bitte zuerst Embeddings erzeugen: python main.py --batch-size 5000"
    exit 1
fi

# Prüfe ob embeddings_map.json existiert
if [ ! -f "$DATA_DIR/embeddings_map.json" ]; then
    echo "❌ Error: $DATA_DIR/embeddings_map.json nicht gefunden!"
    exit 1
fi

# Erstelle releases/ Verzeichnis
mkdir -p "$RELEASE_DIR"

echo "📦 Erstelle Release-Archiv..."
echo ""

# Dateigrößen anzeigen
echo "📊 Dateigrößen:"
echo "   vector.index:        $(du -h $DATA_DIR/vector.index | cut -f1)"
echo "   embeddings_map.json: $(du -h $DATA_DIR/embeddings_map.json | cut -f1)"
echo ""

# Erstelle ZIP mit Fortschrittsanzeige
echo "🗜️  Komprimiere Dateien..."
cd "$DATA_DIR"
zip -9 -r "../${ZIP_FILE}" vector.index embeddings_map.json
cd ..

echo ""
echo "✅ Release-Archiv erstellt!"
echo ""
echo "📁 Archiv-Details:"
echo "   Datei:    $ZIP_FILE"
echo "   Größe:    $(du -h $ZIP_FILE | cut -f1)"
echo ""

# Erstelle README für Release
README_FILE="${RELEASE_DIR}/README_${RELEASE_NAME}.md"
cat > "$README_FILE" << EOF
# RAG Förderkatalog - Pre-Indizierter FAISS-Index v${VERSION}

## 📦 Inhalt

Diese ZIP-Datei enthält den vorbereiteten FAISS-Index für RAG Förderkatalog v${VERSION}.

### Dateien

- \`vector.index\` — FAISS-Index mit Embeddings für alle Projekte
- \`embeddings_map.json\` — Mapping zwischen FAISS-IDs und CSV-Zeilen

## 🚀 Installation

\`\`\`bash
# 1. ZIP herunterladen und entpacken
unzip ${RELEASE_NAME}.zip -d data/

# 2. Prüfen ob Dateien vorhanden sind
ls -lh data/vector.index data/embeddings_map.json

# 3. Anwendung starten
python main.py --no-embeddings
\`\`\`

## ℹ️ Hinweise

- **CSV-Datei benötigt**: Die \`foerderkatalog_export.csv\` muss separat in \`input/\` abgelegt werden
- **Ollama erforderlich**: Installieren Sie Ollama und das Modell \`nomic-embed-text\`
- **GROQ API Key**: Für LLM-Funktionen benötigen Sie einen API-Key in \`.env\`

## 🔐 Checksums

\`\`\`
SHA256: $(sha256sum $ZIP_FILE | cut -d' ' -f1)
MD5:    $(md5sum $ZIP_FILE | cut -d' ' -f1)
\`\`\`

## 📊 Statistiken

- **Anzahl Embeddings**: $(grep -o '"' data/embeddings_map.json | wc -l | awk '{print int($1/2)}')
- **Embedding-Dimension**: 768 (nomic-embed-text)
- **Index-Typ**: FAISS IndexFlatIP (Inner Product)

## 📄 Lizenz

MIT License - siehe Haupt-Repository für Details.

---

**Version**: ${VERSION}
**Erstellt**: $(date +"%Y-%m-%d %H:%M:%S")
**Repository**: https://github.com/dgaida/rag_foerderkatalog
EOF

echo "📝 README erstellt: $README_FILE"
echo ""

# Erstelle Checksums-Datei
CHECKSUM_FILE="${RELEASE_DIR}/${RELEASE_NAME}.sha256"
sha256sum "$ZIP_FILE" > "$CHECKSUM_FILE"

echo "🔐 Checksum-Datei erstellt: $CHECKSUM_FILE"
echo "   SHA256: $(cat $CHECKSUM_FILE)"
echo ""

# Zusammenfassung
echo "════════════════════════════════════════════════════════════"
echo "✅ Release v${VERSION} erfolgreich erstellt!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📋 Release-Dateien:"
echo "   1. $ZIP_FILE"
echo "   2. $README_FILE"
echo "   3. $CHECKSUM_FILE"
echo ""
echo "🚀 Nächste Schritte:"
echo "   1. ZIP-Datei auf GitHub Release hochladen"
echo "   2. README in Release-Notes einfügen"
echo "   3. Git-Tag erstellen: git tag -a v${VERSION} -m 'Release v${VERSION}'"
echo "   4. Tag pushen: git push origin v${VERSION}"
echo ""
echo "💡 Upload-Befehl für GitHub CLI:"
echo "   gh release create v${VERSION} $ZIP_FILE -F RELEASE_NOTES_v${VERSION}.md"
echo ""
