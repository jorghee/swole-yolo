import csv
import math

classes = set()
with open(r'c:\IA\datasets_porciones\dataset_3000\etiquetas_3000.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        target = row[1]
        if target == 'none':
            continue
        objects = target.split(';')
        for obj in objects:
            parts = obj.split()
            if len(parts) >= 6:
                c = int(parts[0])
                classes.add(c)
print(f"Unique classes: {sorted(list(classes))}")
