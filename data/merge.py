import os
import shutil
import uuid

SOURCE_ROOT = "./"
TARGET_ROOT = "organized_dataset"

def normalize(name):
    return name.lower().replace("_", " ").replace("-", " ").strip()

mapping = {
    # Tree
    "tree": "treepose",
    "vriksasana": "treepose",
    "vrikshasana": "treepose",

    # Chair
    "utkatasana": "chairpose",
    "chair": "chairpose",

    # Warrior
    "warrior": "warriorpose",
    "warrior1": "warriorpose",
    "warrior2": "warriorpose",
    "warrior3": "warriorpose",
    "virabhadrasana": "warriorpose",
    "virabhadrasana i": "warriorpose",
    "virabhadrasana ii": "warriorpose",
    "virabhadrasana iii": "warriorpose",

    # Cobra
    "bhujangasana": "cobrapose",
    "cobra": "cobrapose",

    # Downward Dog
    "downdog": "downwarddog",
    "downward dog": "downwarddog",
    "adho mukha svanasana": "downwarddog",

    # Goddess
    "goddess": "goddesspose",
    "utkata konasana": "goddesspose",

    # Corpse
    "savasana": "corpsepose",
    "sivasana": "corpsepose",

    # Bridge
    "bridge": "bridgepose",
    "setu bandha sarvangasana": "bridgepose",

    # Supine Twist
    "supta matsyendrasana": "supinetwist",

    # Happy Baby
    "ananda balasana": "happybabypose"
}

moved = 0
skipped = 0

for root, dirs, files in os.walk(SOURCE_ROOT):
    folder_name = normalize(os.path.basename(root))

    if folder_name not in mapping:
        skipped += 1
        continue

    target_class = mapping[folder_name]
    target_dir = os.path.join(TARGET_ROOT, target_class)
    os.makedirs(target_dir, exist_ok=True)

    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            src_path = os.path.join(root, file)

            new_name = f"{uuid.uuid4().hex}.jpg"
            dst_path = os.path.join(target_dir, new_name)

            shutil.move(src_path, dst_path)
            moved += 1

print(f"Done. Moved: {moved}, Skipped folders: {skipped}")
