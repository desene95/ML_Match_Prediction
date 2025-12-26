import os
import requests

OWNER = "desene95"
REPO = "ML_Match_Prediction"

ASSETS = [
    "football_model.pkl",
    "elo_ratings.pkl",
    "metadata.json",
    "matches_snapshot.parquet",
]

def main():
    os.makedirs("models", exist_ok=True)

    rel = requests.get(
        f"https://api.github.com/repos/{OWNER}/{REPO}/releases/latest",
        timeout=30,
    )
    rel.raise_for_status()
    rel_json = rel.json()

    asset_map = {a["name"]: a["browser_download_url"] for a in rel_json["assets"]}

    for name in ASSETS:
        url = asset_map.get(name)
        if not url:
            raise SystemExit(f"Missing asset in latest release: {name}")

        r = requests.get(url, timeout=120)
        r.raise_for_status()

        out_path = os.path.join("models", name)
        with open(out_path, "wb") as f:
            f.write(r.content)

    print("Downloaded:", ", ".join(ASSETS))

if __name__ == "__main__":
    main()
