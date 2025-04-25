from utils import fetch_json, save_json

COMFYUI_MANAGER_LIST_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json"


def fetch_custom_node_list():
    return fetch_json(COMFYUI_MANAGER_LIST_URL)


def save_custom_node_list():
    data = fetch_custom_node_list()
    save_json(data, 'src/data/custom-node-list.json')
    print("custom-node-list.json saved.")


if __name__ == "__main__":
    save_custom_node_list()
