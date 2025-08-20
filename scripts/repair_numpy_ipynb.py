import json, shutil, sys, os

P = r'd:\CODES\Machine Learning\Notebooks\PRE\numpy.ipynb'

with open(P, 'r', encoding='utf-8') as f:
    outer = json.load(f)

cells = outer.get('cells', [])
if not cells or not isinstance(cells, list):
    print('No cells found; aborting to avoid data loss.')
    sys.exit(2)

first = cells[0]
if not isinstance(first, dict):
    print('Unexpected first cell; aborting.')
    sys.exit(3)

src = first.get('source')
if not isinstance(src, list) or not src:
    print('First cell has no source; aborting.')
    sys.exit(4)

inner_text = ''.join(src).strip()
if not (inner_text.startswith('{') and inner_text.endswith('}')):
    print('Embedded notebook JSON not detected; aborting.')
    sys.exit(5)

try:
    inner_nb = json.loads(inner_text)
except Exception as e:
    print('Failed to parse embedded JSON:', e)
    sys.exit(6)

for key in ('cells','metadata','nbformat'):
    if key not in inner_nb:
        print('Embedded JSON missing key:', key)
        sys.exit(7)

backup = P + '.bak'
shutil.copy2(P, backup)
with open(P, 'w', encoding='utf-8') as f:
    json.dump(inner_nb, f, ensure_ascii=False, indent=1)

print('Repaired notebook written. Backup at:', backup)
print('Cells:', len(inner_nb.get('cells', [])), 'nbformat:', inner_nb.get('nbformat'))
