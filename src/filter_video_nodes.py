import json
import os

INPUT_PATH = "src/data/custom-node-list.json"
OUTPUT_PATH = "src/data/video-node-list.json"


def is_video_node(node):
    for key in ["description", "title"]:
        value = node.get(key, "")
        if isinstance(value, str) and "video" in value.lower():
            return True
    return False


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("custom_nodes", data) if isinstance(data, dict) else data
    video_nodes = [node for node in nodes if is_video_node(node)]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(video_nodes, f, indent=2, ensure_ascii=False)
    print(f"Found {len(video_nodes)} video nodes. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
