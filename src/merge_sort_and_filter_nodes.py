"""
Merges a list of custom ComfyUI nodes with their GitHub statistics,
filters out nodes related to "video", sorts them by star count in
descending order, and saves the result.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CUSTOM_NODES_FILE = os.path.join(DATA_DIR, "custom-node-list.json")
GITHUB_STATS_FILE = os.path.join(DATA_DIR, "github-stats.json")
# Changed output filename to reflect filtering
OUTPUT_FILE = os.path.join(DATA_DIR, "filtered_sorted_custom_nodes.json")


def load_json_file(file_path):
    """Loads and parses a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON file {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
    return None


def normalize_repo_url(url_string):
    """Normalizes a GitHub repository URL by removing a trailing '.git' if present."""
    if not isinstance(url_string, str):
        return None
    if url_string.endswith(".git"):
        return url_string[:-4]
    return url_string


def merge_and_sort_nodes():
    """
    Merges the custom node list with GitHub stats, filters to keep only video-related nodes,
    and sorts by star count.
    """
    custom_node_data = load_json_file(CUSTOM_NODES_FILE)
    github_stats_data = load_json_file(GITHUB_STATS_FILE)

    if not custom_node_data or not github_stats_data:
        print("Error: Failed to load input files.")
        return

    nodes_list = custom_node_data.get("custom_nodes", [])

    filtered_nodes_list = []
    for node in nodes_list:
        title = node.get("title", "").lower()
        reference_url = node.get("reference", "")
        # Normalize reference URL before checking for "video" keyword
        reference_lower = (
            normalize_repo_url(reference_url).lower() if reference_url else ""
        )
        description = node.get("description", "").lower()

        # Corrected filtering logic:
        # Keep the node if "video" IS in title OR IS in reference OR IS in description.
        # This means nodes where "video" is NOT present in any of these fields will be excluded.
        if "video" in title or "video" in reference_lower or "video" in description:
            filtered_nodes_list.append(node)
            # Optional: print if a node is kept
            # print(f"Keeping node (contains 'video'): Title='{node.get('title', 'N/A')}', Ref='{reference_url}'")
        else:
            print(
                f"Filtering out node (does not contain 'video'): Title='{node.get('title', 'N/A')}', Ref='{reference_url}'"
            )

    processed_nodes = []
    # Iterate over the correctly filtered list
    for node in filtered_nodes_list:
        current_node = node.copy()

        repo_url_from_node = current_node.get("reference")
        normalized_url_for_lookup = normalize_repo_url(repo_url_from_node)

        stars = 0
        author_account_age_days = 0  # Default value

        if normalized_url_for_lookup and normalized_url_for_lookup in github_stats_data:
            stats = github_stats_data[normalized_url_for_lookup]
            stars = stats.get("stars", 0)
            author_account_age_days = stats.get(
                "author_account_age_days", 0
            )  # Get author_account_age_days
        elif normalized_url_for_lookup:
            # This case handles when the repo URL is present but not found in github_stats_data
            # stars and author_account_age_days will remain 0 as initialized
            pass

        current_node["stars"] = stars
        current_node["author_account_age_days"] = (
            author_account_age_days  # Add to current_node
        )
        processed_nodes.append(current_node)

    processed_nodes.sort(key=lambda x: x.get("stars", 0), reverse=True)

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"custom_nodes": processed_nodes}, f, ensure_ascii=False, indent=2
            )
        # Updated success message with new filename
        print(
            f"Successfully merged, filtered, and sorted nodes. Output saved to {OUTPUT_FILE}"
        )
    except IOError as e:
        print(f"Error saving output file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving output: {e}")


if __name__ == "__main__":
    merge_and_sort_nodes()
