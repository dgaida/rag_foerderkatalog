import requests
from tqdm import tqdm

INDEX_URL = "https://github.com/dgaida/rag-foerderkatalog/releases/download/v1.0/index.zip"


def download_index():
    print("📥 Lade vorberechneten Index herunter...")
    response = requests.get(INDEX_URL, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open("index.zip", "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print("📦 Entpacke Index...")
    import zipfile

    with zipfile.ZipFile("index.zip", "r") as zip_ref:
        zip_ref.extractall("data/")

    print("✅ Index bereit!")


if __name__ == "__main__":
    download_index()
