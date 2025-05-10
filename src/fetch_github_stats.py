from utils import fetch_json, save_json

GITHUB_STATS_URL = (
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/github-stats.json"
)


def fetch_github_stats():
    return fetch_json(GITHUB_STATS_URL)


def save_github_stats():
    data = fetch_github_stats()
    save_json(data, "src/data/github-stats.json")
    print("github-stats.json saved.")


if __name__ == "__main__":
    save_github_stats()
