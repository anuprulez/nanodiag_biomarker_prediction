#!/usr/bin/env python3
"""Extract CpG ID (C), Gene (D), and signed delta beta (V) from MOESM3."""
import csv
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'data/41375_2021_1316_MOESM3_ESM.xlsx'
OUTPUT = BASE / 'data/aml_cpg_gene_delta_beta.tsv'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def extract(source=SOURCE, output=OUTPUT):
    with zipfile.ZipFile(source) as archive:
        strings = [''.join(item.itertext()) for item in
                   ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        rows = []
        headers = None
        for row in sheet.findall('s:sheetData/s:row', NS):
            values = {}
            for cell in row:
                column = re.sub(r'\d+', '', cell.attrib['r'])
                value = cell.find('s:v', NS)
                text = value.text if value is not None else ''
                if cell.get('t') == 's':
                    text = strings[int(text)]
                values[column] = text
            if row.get('r') == '2':
                headers = values
            if re.fullmatch(r'cg\d+', values.get('C', '')):
                float(values['V'])  # Ensure the requested measurement is numeric.
                rows.append([values['C'], values.get('D', ''), values['V'], values['C'] + '_' + values.get('D', '')])
    if not headers or [headers.get(c) for c in ('C', 'D', 'V')] != [
            'CpG ID', 'Gene', 'GSE50660 vs GSE58477']:
        raise ValueError('Workbook columns differ from expected supplementary table')
    if not rows or len({row[0] for row in rows}) != len(rows):
        raise ValueError('Missing or duplicate CpG identifiers')
    with output.open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(['CpG_ID', 'Gene', 'deltaBeta_GSE50660_vs_GSE58477', 'CpG_Gene'])
        writer.writerows(rows)
    print(f'Extracted {len(rows)} CpGs to {output}')


if __name__ == '__main__':
    extract()
