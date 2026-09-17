"""Turn selected run directories into a reviewable JSON evidence report.

This reporting pipeline reads existing measurements; it never launches training
or restores a process. Raw images and logs remain private in ignored runs/.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import metrics
from prepare import file_hash, write_json


def main():
    """Reanalyze selected trials, preserve provenance, and summarize passed timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--controller-clock-offset-ns', type=int,
                        help='Required only to reanalyze older runs that did not record the controller clock offset')
    args = parser.parse_args()
    # Analysis source can change after training (for example, a clock correction).
    # Record those hashes separately from the code hashes in each original key.
    report = {'scope': 'same-host, warm cached assets; numerical success is separate from compatibility',
              'keys': {}, 'runs': {}, 'timing_summary': {},
              'analysis_source_sha256': {name: file_hash(Path(__file__).parent / name)
                                         for name in ('metrics.py', 'prepare.py', 'report.py')}}
    for run in args.runs:
        key = json.loads((run / 'key.json').read_text())
        # Deduplicate identical environment/configuration records by stable digest.
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        report['keys'][digest] = key
        result = json.loads((run / 'result.json').read_text())
        if 'controller_monotonic_offset_ns' not in result and args.controller_clock_offset_ns is None:
            raise ValueError('Older runs require the measured --controller-clock-offset-ns for reanalysis')
        original = result.get('latencies')
        metrics.finish(result, run, args.controller_clock_offset_ns)
        if original != result['latencies']:
            # Keep the superseded calculation so a correction is auditable.
            result['superseded_latencies'] = original
        result['key_sha256'] = digest
        # Publish fingerprints of state records, not their full tensor contents.
        result['state_file_sha256'] = {p.name: file_hash(p) for p in sorted(run.glob('state-*.json'))}
        report['runs'][run.name] = result
    for mode in ('application', 'criu'):
        # Diagnostic trials include extra inspection work and must not enter the
        # headline timing summary. The caller supplies matching paired trials.
        trials = [r for r in report['runs'].values() if r['mode'] == mode and r['timing']]
        if not trials:
            continue
        if not all(r['lifecycle_passed'] and r['numerical_passed'] for r in trials):
            raise ValueError('Failed timing trial; inspect individual evidence before reporting statistics')
        summary = {}
        for metric in ('capture_to_sync_seconds', 'restore_to_next_update_seconds',
                       'capture_to_gpu_observed_seconds', 'job_b_seconds'):
            # Headline trials contain one capture; repeated-capture runs are diagnostic.
            values = [r['latencies'][0][metric] for r in trials]
            summary[metric] = {'individual': values, 'median': statistics.median(values),
                               'min': min(values), 'max': max(values)}
        report['timing_summary'][mode] = summary
    write_json(args.output, report)
    print(json.dumps(report['timing_summary'], indent=2))


if __name__ == '__main__':
    main()
