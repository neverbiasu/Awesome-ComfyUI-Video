import requests
import json
import os


def fetch_json(url):
    resp = requests.get(url)
    if resp.status_code == 200:
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            print(f"Invalid JSON response from {url}: {e}")
            print(f"Response content: {resp.text[:100]}...")
            return {}
    else:
        raise Exception(f"Failed to fetch {url}: {resp.status_code}")


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fetch_github_file(repo_url, filepath):
    try:
        if repo_url.endswith('/'):
            repo_url = repo_url[:-1]
        parts = repo_url.split('/')
        owner, repo = parts[-2], parts[-1]
        # Try main branch first
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{filepath}"
        resp = requests.get(raw_url)
        if resp.status_code == 200:
            return resp.text
        # Try master branch
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{filepath}"
        resp = requests.get(raw_url)
        if resp.status_code == 200:
            return resp.text
        return None
    except Exception as e:
        print(f"Error fetching {filepath} from {repo_url}: {e}")
        return None
