import json, re
from pathlib import Path
from collections import defaultdict

raw_dir = Path('results/raw/hf_local/llama3-8b/z_shot_local/seed2025')
json_files = sorted(f for f in raw_dir.glob('*.json')
                    if 'metadata' not in f.name)

groups = defaultdict(list)
for f in json_files:
    w = json.loads(f.read_text())
    groups[w['group_id']].append(w)

header = f"{'CHUNK':<25} {'EXP':>4} {'FOUND':>5} {'PARSEABLE':>9} {'STARTS_[':>8} {'ENDS_]':>6} {'STATUS':<14}"
print(header)
print('-' * 80)

total_exp = total_found = total_ok = 0

for gid in sorted(groups):
    for w in sorted(groups[gid], key=lambda x: x['chunk_index']):
        resp      = w['response_text']
        expected  = len(w['token_indices'])
        chunk_id  = w['chunk_id']
        starts_ok = resp.lstrip().startswith('[')
        ends_ok   = resp.rstrip().endswith(']')

        records = re.findall(r'{"index"', resp)
        found   = len(records)

        try:
            first     = resp.find('[')
            last      = resp.rfind(']')
            arr       = json.loads(resp[first:last+1]) if first != -1 else []
            parseable = len(arr)
        except Exception:
            parseable = 0

        if found == expected and parseable == expected and starts_ok and ends_ok:
            status = 'OK'
        elif parseable == expected:
            status = 'OK (preamble)'
        elif parseable > 0:
            status = f'PARTIAL ({parseable}/{expected})'
        else:
            status = 'FAIL'

        row = f"{chunk_id:<25} {expected:>4} {found:>5} {parseable:>9} {str(starts_ok):>8} {str(ends_ok):>6} {status:<14}"
        print(row)
        total_exp   += expected
        total_found += found
        total_ok    += parseable

print('-' * 80)
print(f"{'TOTAL':<25} {total_exp:>4} {total_found:>5} {total_ok:>9}")
print()
pct = 100 * total_ok / total_exp if total_exp > 0 else 0
print(f"Overall completeness: {total_ok}/{total_exp} tokens parseable ({pct:.1f}%)")