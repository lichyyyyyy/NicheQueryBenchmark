import json
from typing import Dict, Tuple


def parse_parcellation_structure(json_path: str) -> Dict[int, Tuple[int, str, int]]:
    """
    Parse parcellation structure and return a dictionary:

    key: structure id
    value: (parent_id, name, st_level)
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    result = {}

    def traverse(node: dict):
        """
        Recursively traverse the tree and extract required fields.
        """
        structure_id = node["id"]
        parent_id = node["parent_structure_id"]
        name = node["name"]
        st_level = node["st_level"]

        result[structure_id] = {
            'parent_id': parent_id,
            'name': name,
            'st_level': st_level,
        }

        # Recursively process children
        for child in node.get("children", []):
            traverse(child)

    # Root is inside data["msg"]
    for root_node in data["msg"]:
        traverse(root_node)

    return result