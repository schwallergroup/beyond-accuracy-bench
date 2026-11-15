import csv
import json
from pathlib import Path
from bench.judge.trace_fetch import fetch_trace_record
from bench.judge.trace_io import load_trace_record, save_trace_record, trace_slug

base = Path(__file__).resolve().parent
rows = list(csv.DictReader((base / 'manifest_iteration2_4.csv').open()))
traces = base / 'traces'
missing = []
for index, row in enumerate(rows, 1):
    path = traces / (trace_slug(row) + '.json')
    if path.exists():
        load_trace_record(traces, row)
        continue
    try:
        record = fetch_trace_record(row['tag'], row['sample_id'], int(row['iteration']), row['model_label'])
        save_trace_record(record, path)
    except ValueError as exc:
        missing.append({**row, 'error': str(exc)})
    if index % 25 == 0:
        print(f'Processed {index}/{len(rows)}; unavailable {len(missing)}', flush=True)
with (base / 'unavailable_traces.json').open('w') as f:
    json.dump(missing, f, indent=2)
print(f'Finished; unavailable {len(missing)}', flush=True)
