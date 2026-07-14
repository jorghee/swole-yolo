import os
from collections import Counter
import yaml

def main():
    train_dir = r'c:\IA\yolo_obb_dataset\labels\train'
    yaml_path = r'dataset.yaml'
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {})
    
    c = Counter()
    for f in os.listdir(train_dir):
        if f.endswith('.txt'):
            with open(os.path.join(train_dir, f), 'r') as file:
                for line in file:
                    cls_id = int(line.split()[0])
                    c[cls_id] += 1
                    
    print("Class distribution in training set:")
    for cls_id in sorted(c.keys()):
        cls_name = names.get(cls_id, str(cls_id))
        print(f"Class {cls_id} ({cls_name}): {c[cls_id]} instances")

if __name__ == '__main__':
    main()
