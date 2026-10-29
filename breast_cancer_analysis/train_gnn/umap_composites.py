"""Compose archived UMAP PDFs without refitting or rasterizing their contents."""
import argparse
import json
from pathlib import Path
import warnings


def select_umap_pairs(runs, model='PNA', feature_type='PNAConv4'):
    pairs = []
    for run in sorted(map(Path, runs), key=lambda p: int(p.name[3:])):
        before = list((run/'generated').rglob(f'UMAP_nedbit_dnam_features_test_before_GNN_{model}.pdf'))
        after = list((run/'generated').rglob(f'UMAP_node_embeddings_*_{feature_type}_{model}.pdf'))
        if len(before) > 1 or len(after) > 1:
            raise ValueError(f'{run}: ambiguous UMAP sources for {model}/{feature_type}')
        if not before or not after:
            warnings.warn(f'{run.name}: missing before/after {model} UMAP; skipping composites')
            return []
        pairs.append((run.name, before[0], after[0]))
    if len(pairs) != 5:
        warnings.warn(f'Five UMAP runs required, found {len(pairs)}; composites skipped')
        return []
    return pairs


def publication_umap_page(document, tick_scale=1.4):
    """Enlarge archived Matplotlib ticks in memory, retaining vector minus glyphs.

    Scale each complete text group about its tick anchor, including the separate
    minus-sign form used by Matplotlib. Source archives remain untouched.
    """
    import re
    page = document[0]
    content = page.read_contents().decode('latin1')
    number = r'[-+0-9.eE]+'
    clip = re.search(r'q ('+number+r') ('+number+r') ('+number+r') ('+number+r') re W n', content)
    if not clip:
        return
    left, bottom, width, height = map(float, clip.groups())
    pattern = re.compile(r'q\n(?P<a>'+number+r') (?P<b>'+number+r') (?P<c>'+number+r') '
                         r'(?P<d>'+number+r') (?P<x>'+number+r') (?P<y>'+number+r') cm\n'
                         r'BT\n(?P<body>.*?)\nET\n(?P<minus>q [^\n]*minus Do Q\n)?Q', re.S)
    matches = list(pattern.finditer(content))
    ticks = []
    for match in matches:
        body = match['body']
        x, y = float(match['x']), float(match['y'])
        font = re.search(r'/\S+ ('+number+r') Tf', body)
        if not font:
            continue
        if 'UMAP1' in body or 'UMAP2' in body or (y > bottom+height and float(font[1]) >= 18):
            # Remove the repeated per-panel source title.
            ticks.append((match, 'title'))
        elif abs(float(match['a'])-1)<1e-6 and abs(float(match['d'])-1)<1e-6:
            if x >= left and y < bottom and not 'UMAP' in body:
                ticks.append((match, 'x'))
            elif x < left and bottom < y < bottom+height and not 'UMAP' in body:
                ticks.append((match, 'y'))
    x_ticks = [m for m, kind in ticks if kind == 'x']
    replacements = []
    for match, kind in ticks:
        if kind == 'title' or (kind == 'x' and len(x_ticks)>6 and x_ticks.index(match)%2):
            replacements.append((match.start(), match.end(), ''))
            continue
        # Numeric spans include their font metrics; account for an outlined minus.
        x, y = float(match['x']), float(match['y'])
        spans = [s for block in page.get_text('dict')['blocks']
                 for line in block.get('lines', []) for s in line['spans']
                 if abs(s['origin'][1]-(page.rect.height-y)) < .1
                 and x-.1 <= s['origin'][0] < x+50]
        right = max((s['bbox'][2] for s in spans), default=x+20)
        anchor_x = (x+right)/2 if kind == 'x' else right
        anchor_y = y
        transform = (f'q\n{tick_scale} 0 0 {tick_scale} '
                     f'{(1-tick_scale)*anchor_x:.6f} {(1-tick_scale)*anchor_y:.6f} cm\n')
        replacements.append((match.start(), match.end(), transform+match[0]+'\nQ'))
    for start, end, replacement in reversed(replacements):
        content = content[:start]+replacement+content[end:]
    page.clean_contents()
    document.update_stream(page.get_contents()[0], content.encode('latin1'))


def plot_umap_run_composites(runs, output, dpi=300, model='PNA', feature_type='PNAConv4'):
    """Create two five-panel figures with the same runs ordered numerically.

    Existing vector PDFs retain their original labels, coordinates and colors.
    Each run has an independent UMAP coordinate system; coordinates are not pooled.
    """
    pairs = select_umap_pairs(runs, model, feature_type)
    if not pairs:
        return []
    try:
        import pymupdf
    except ModuleNotFoundError as error:
        if error.name != 'pymupdf':
            raise
        requirements = Path(__file__).with_name('requirements-plotting.txt')
        warnings.warn(
            'UMAP composites skipped because PyMuPDF is unavailable. '
            f'Install it in the active environment with: python -m pip install -r {requirements}. '
            'Other accumulation outputs will continue.',
            stacklevel=2,
        )
        return []
    output = Path(output)
    plots = output/'plots'
    plots.mkdir(parents=True, exist_ok=True)
    paths = []
    for stage, source_index, description in (
            ('before', 1, 'Raw NeDBIT + DNA methylation features'),
            ('after', 2, 'After PNA training: five independent runs')):
        with pymupdf.open() as document:
            page = document.new_page(width=1296, height=864)
            page.insert_textbox(pymupdf.Rect(30, 12, 1266, 64),
                                ('Breast Cancer: Trained last PNAConv node embeddings' if stage == 'after'
                                 else 'Breast Cancer: UMAP before PNA training'),
                                fontsize=26, fontname='helv', align=1)
            page.insert_textbox(pymupdf.Rect(30, 65, 1266, 101), description,
                                fontsize=18, fontname='helv', align=1)
            for index, pair in enumerate(pairs):
                row, col = divmod(index, 3)
                # Center the two panels in the second row.
                x = 24 + col*416 + (208 if row else 0)
                y = 105 + row*360
                page.insert_textbox(pymupdf.Rect(x, y, x+400, y+42),
                                    f'{chr(65+index)}. Run {pair[0][3:]}',
                                    fontsize=21, fontname='hebo', align=1)
                with pymupdf.open(pair[source_index]) as source:
                    if len(source) != 1:
                        raise ValueError(f'{pair[source_index]}: expected one UMAP page')
                    publication_umap_page(source)
                    page.show_pdf_page(pymupdf.Rect(x, y+43, x+400, y+348), source, 0,
                                       keep_proportion=True)
                page.insert_textbox(pymupdf.Rect(x+65, y+331, x+390, y+359),
                                    'UMAP 1', fontsize=16, fontname='helv', align=1)
                page.insert_text((x+10, y+237), 'UMAP 2', fontsize=16,
                                 fontname='helv', rotate=90)
            stem = plots/f'UMAP_{stage}_{model}_5_runs'
            pdf = stem.with_suffix('.pdf')
            document.save(pdf, garbage=4, deflate=True)
            page.get_pixmap(dpi=max(300, int(dpi)), alpha=False).save(stem.with_suffix('.png'))
            paths.extend([pdf, stem.with_suffix('.png')])
    data = output/'data'
    data.mkdir(parents=True, exist_ok=True)
    (data/f'UMAP_{model}_5_runs_sources.json').write_text(json.dumps(dict(
        model=model, after_layer=feature_type,
        method='Vector PDF composition of existing UMAPs; enlarged ticks; no refitting or pooling',
        runs=[dict(run=name, before=str(before.resolve()), after=str(after.resolve()))
              for name, before, after in pairs]), indent=2)+'\n')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    runs = [p for p in (args.dataset/'runs').glob('run*')
            if p.is_dir() and p.name[3:].isdigit()]
    for path in plot_umap_run_composites(runs, args.dataset/'accumulated_results', args.dpi):
        print(path)


if __name__ == '__main__':
    main()
