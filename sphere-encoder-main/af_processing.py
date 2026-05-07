import os
import json
from glob import glob

root = "workspace/datasets/animal-faces"

classes = {
    "cat": 0,
    "dog": 1,
    "wild": 2,
}

for split in ["train", "val"]:

    entries = []

    for class_name, class_id in classes.items():

        pattern = os.path.join(
            root,
            split,
            class_name,
            "*"
        )

        image_paths = glob(pattern)

        for path in image_paths:

            rel_path = os.path.relpath(path, root)

            entries.append({
                "image_path": rel_path,
                "class_id": class_id,
                "class_name": class_name,
            })

    output_path = os.path.join(root, f"{split}.json")

    with open(output_path, "w") as f:

        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(entries)} entries to {output_path}")