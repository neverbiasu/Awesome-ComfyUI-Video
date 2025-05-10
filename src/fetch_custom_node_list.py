from utils import fetch_json, save_json

CUSTOM_NODE_LIST_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json"


def fetch_custom_node_list():
    """Fetches the custom node list from the specified URL."""
    return fetch_json(CUSTOM_NODE_LIST_URL)


def save_custom_node_list():
    """Fetches the custom node list and saves it to a local JSON file."""
    data = fetch_custom_node_list()
    save_json(data, "src/data/custom-node-list.json")
    print("custom-node-list.json saved.")


if __name__ == "__main__":
    # Main execution block: saves the custom node list when the script is run directly.
    save_custom_node_list()
