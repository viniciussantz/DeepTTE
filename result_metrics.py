import argparse
import math
import statistics


def parse_result_file(path):
    rows = []
    skipped = 0

    with open(path, 'r') as f:
        first_line = f.readline()
        if not first_line:
            return rows, skipped

        first = first_line.strip().split()
        has_header = len(first) > 0 and first[0] == 'source_file'

        if not has_header:
            try:
                label = float(first[-2])
                pred = float(first[-1])
                rows.append((label, pred))
            except (ValueError, IndexError):
                skipped += 1

        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                label = float(parts[-2])
                pred = float(parts[-1])
                rows.append((label, pred))
            except (ValueError, IndexError):
                skipped += 1

    return rows, skipped


def summarize(rows):
    finite = [(y, p) for y, p in rows if math.isfinite(y) and math.isfinite(p)]
    total = len(rows)
    finite_n = len(finite)

    if finite_n == 0:
        return {
            'n_total': total,
            'n_finite': 0,
            'label_mean': float('nan'),
            'pred_mean': float('nan'),
            'mae': float('nan'),
            'rmse': float('nan'),
            'mape_pct': float('nan'),
            'bias': float('nan'),
            'p50_abs_error': float('nan')
        }

    labels = [y for y, _ in finite]
    preds = [p for _, p in finite]
    errors = [p - y for y, p in finite]
    abs_errors = [abs(e) for e in errors]

    mae = sum(abs_errors) / finite_n
    rmse = math.sqrt(sum(e * e for e in errors) / finite_n)
    mape = 100.0 * sum(abs(p - y) / max(abs(y), 1e-9) for y, p in finite) / finite_n
    bias = sum(errors) / finite_n

    return {
        'n_total': total,
        'n_finite': finite_n,
        'label_mean': sum(labels) / finite_n,
        'pred_mean': sum(preds) / finite_n,
        'mae': mae,
        'rmse': rmse,
        'mape_pct': mape,
        'bias': bias,
        'p50_abs_error': statistics.median(abs_errors)
    }


def summarize_by_duration(rows):
    finite = [(y, p) for y, p in rows if math.isfinite(y) and math.isfinite(p)]
    
    bins = [
        ('<=10min', 0, 600),        
        ('10-20min', 600, 1200),
        ('20-40min', 1200, 2400),
        ('40-80min', 2400, 4800),
        ('>80min', 4800, float('inf')),
    ]

    out = []
    for name, lo, hi in bins:
        subset = [(y, p) for y, p in finite if (y > lo and y <= hi)]
        if not subset:
            out.append((name, 0, float('nan'), float('nan'), float('nan')))
            continue

        n = len(subset)
        mae = sum(abs(p - y) for y, p in subset) / n
        mape = 100.0 * sum(abs(p - y) / max(abs(y), 1e-9) for y, p in subset) / n
        bias = sum((p - y) for y, p in subset) / n
        out.append((name, n, mae, mape, bias))

    return out


def print_summary(path, metrics, skipped, by_bin):
    print(f'\n=== {path} ===')
    print(f"samples(total/finite): {metrics['n_total']}/{metrics['n_finite']} | skipped_lines: {skipped}")
    print(f"label_mean: {metrics['label_mean']:.6f}")
    print(f"pred_mean: {metrics['pred_mean']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAPE(%): {metrics['mape_pct']:.6f}")
    print(f"Bias(pred-label): {metrics['bias']:.6f}")
    print(f"P50 |error|: {metrics['p50_abs_error']:.6f}")

    print('\nBy duration:')
    print('bin\tn\tMAE\tMAPE(%)\tBias')
    for name, n, mae, mape, bias in by_bin:
        if n == 0:
            print(f'{name}\t0\t-\t-\t-')
        else:
            print(f'{name}\t{n}\t{mae:.6f}\t{mape:.6f}\t{bias:.6f}')


def main():
    parser = argparse.ArgumentParser(description='Compute metrics from DeepTTE result files.')
    parser.add_argument('files', nargs='+', help='Result file(s) with label/pred columns.')
    args = parser.parse_args()

    for path in args.files:
        rows, skipped = parse_result_file(path)
        metrics = summarize(rows)
        by_bin = summarize_by_duration(rows)
        print_summary(path, metrics, skipped, by_bin)


if __name__ == '__main__':
    main()
