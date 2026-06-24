import nbformat

with open('notebooks/01_data_preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.source = cell.source.replace('Helsinki-NLP/opus-mt-en-bn', 'monirbishal/en-bn-nmt')

with open('notebooks/01_data_preprocessing.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
