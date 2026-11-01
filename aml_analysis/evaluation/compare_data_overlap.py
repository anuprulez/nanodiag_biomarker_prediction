"""Compare three parameter sweeps separately for train, test, and validation."""
import argparse
import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

SPLITS = ('train', 'test', 'val')
PARAMETERS = ('corr', 'n_neighbours', 'num_negatives')
COMPARISONS = (
    ('n_neighbours', {'corr': 0.33, 'num_negatives': 10000}),
    ('num_negatives', {'corr': 0.33, 'n_neighbours': 3000}),
    ('corr', {'n_neighbours': 3000, 'num_negatives': 10000}),
)


def load_splits(folder):
    """Read final retained graph nodes; local numeric IDs are not comparable."""
    result = {}
    for split in SPLITS:
        path = folder / 'preprocessed' / f'out_genes_aml_{split}.csv'
        names = set()
        with path.open() as source:
            for number, line in enumerate(source, 1):
                fields = line.split()
                if len(fields) != 3:
                    raise ValueError(f'{path}:{number}: expected ID, feature name, label')
                names.add(fields[1])
        result[split] = names
    return result


def membership_regions(sets):
    """Disjoint regions, so each identifier contributes exactly once to a pie."""
    membership = defaultdict(list)
    for label, values in sets.items():
        for value in values:
            membership[value].append(label)
    return Counter(tuple(labels) for labels in membership.values())


def plot_bars(panels, title, path, varying):
    """Show exclusive membership counts across parameter settings for each split."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 7), squeeze=False)
    # Use the same region order and colors for all three splits.
    ordered = sorted(set().union(*(membership_regions(sets) for sets in panels.values())),
                     key=lambda labels: (len(labels), labels))
    labels = [' & '.join(label.split('=')[1] for label in region) for region in ordered]
    colors = plt.cm.tab20.colors[:len(ordered)]
    for ax, (name, sets) in zip(axes[0], panels.items()):
        regions = membership_regions(sets)
        total = sum(regions.values())
        counts = [regions[region] for region in ordered]
        bars = ax.barh(range(len(ordered)), counts, color=colors)
        ax.set_yticks(range(len(ordered)), labels)
        ax.invert_yaxis()
        for bar, count in zip(bars, counts):
            percent = 100 * count / total if total else 0
            ax.text(count + max(counts, default=0) * .015, bar.get_y() + bar.get_height()/2,
                    f'{count:,} ({percent:.2f}%)', va='center', fontsize=9)
        ax.set_xlim(0, max(max(counts, default=0) * 1.38, 1))
        ax.set_xlabel('Distinct probe–gene IDs')
        ax.set_ylabel(f'Present in exactly these {varying} settings')
        ax.set_title(f'{name} (union: {total:,} IDs)')
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(title + '\nExclusive membership regions; percentages use each split’s union', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def compare(folders, output):
    from run_pipeline import PATTERN
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    skipped = []
    for folder in folders:
        match = PATTERN.fullmatch(folder.name)
        if not match:
            skipped.append((folder.name, 'Non-standard parameter folder name'))
            continue
        params = dict(zip(PARAMETERS, (float(match[1]), int(match[2]), int(match[3]))))
        if not any(all(params[key] == value for key, value in fixed.items())
                   for _, fixed in COMPARISONS):
            continue
        try:
            splits = load_splits(folder)
        except FileNotFoundError as error:
            skipped.append((folder.name, str(error)))
            continue
        datasets[folder.name] = (dict(zip(PARAMETERS, (float(match[1]), int(match[2]), int(match[3])))), splits)
    if not datasets:
        raise ValueError('No complete dataset folders with train/test/val node files')
    rows = []
    pairs = []

    def record(kind, group, panel, sets):
        regions = membership_regions(sets)
        total = sum(regions.values())
        for region, count in sorted(regions.items()):
            rows.append(dict(kind=kind, group=group, split=panel, membership=' & '.join(region),
                             count=count, union=total, percent=100 * count / total))
        for (left, a), (right, b) in combinations(sets.items(), 2):
            union = len(a | b)
            pairs.append(dict(kind=kind, group=group, split=panel, left=left, right=right,
                              left_count=len(a), right_count=len(b), intersection=len(a & b),
                              union=union, jaccard=len(a & b) / union if union else 0))

    groups_count = 0
    for varying, constraints in COMPARISONS:
        fixed = tuple(p for p in PARAMETERS if p != varying)
        groups = defaultdict(list)
        for name, (params, splits) in datasets.items():
            if not all(params[key] == value for key, value in constraints.items()):
                continue
            groups[tuple(params[p] for p in fixed)].append((params[varying], name, splits))
        for values, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            group = '__'.join(f'{p}_{v:g}' for p, v in zip(fixed, values))
            panels = {split: {f'{varying}={value:g}': splits[split]
                              for value, _, splits in sorted(members)} for split in SPLITS}
            for split, sets in panels.items():
                record('across_datasets', group, split, sets)
            plot_bars(panels, f'Vary {varying}; fixed {group}', output / f'vary_{varying}__{group}.png', varying)
            groups_count += 1
    for filename, data in [('membership_regions.csv', rows), ('pairwise_overlap.csv', pairs)]:
        with (output / filename).open('w', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with (output / 'skipped_datasets.csv').open('w', newline='') as target:
        writer = csv.writer(target); writer.writerow(['folder', 'reason']); writer.writerows(skipped)
    print(f'Compared {len(datasets)} datasets, {groups_count} parameter groups; skipped {len(skipped)}. Output: {output}')


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folders', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.folders, args.output)


if __name__ == '__main__':
    main()
