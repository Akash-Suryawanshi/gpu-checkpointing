"""Measure memory and elapsed time in the controller, which manages the trainer.

Monotonic clocks count elapsed time, not calendar dates. Restored clocks may have
different offsets: raw events -> finish() aligns clocks -> latencies() measures.
Keep JSON field names stable for evidence readers.
"""

import json
from pathlib import Path
import re
import threading


def sample_memory(get_pid):
    """Sample memory; return a function that stops sampling and collects the result.

    get_pid follows the trainer's process ID across restarts. Linux's /proc files
    expose process memory. Cgroups group processes to count and limit their memory,
    including cached file data.
    """
    # This host uses cgroup v2: `0::/path` identifies the controller's group.
    group_path = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]
    group = Path('/sys/fs/cgroup') / group_path.lstrip('/')
    limits = {}
    # Ancestor limits apply even when this group's "max" means no local limit.
    for parent in (group, *group.parents):
        if parent == Path('/sys/fs'):
            break
        if (parent / 'memory.max').exists():
            limits[str(parent)] = (parent / 'memory.max').read_text().strip()
    usage = {'sampling_interval_seconds': 0.1, 'process_peak_rss_bytes': 0,
             'process_peak_hwm_bytes': 0, 'cgroup_peak_bytes': 0,
             'host_min_available_bytes': None, 'cgroup': str(group), 'cgroup_limits': limits}
    # A thread runs alongside the controller's main code; Event requests its stop.
    stop = threading.Event()

    def sample():
        """Collect process, cgroup, and host observations until the stop event."""
        while not stop.is_set():
            try:
                rows = Path(f'/proc/{get_pid()}/status').read_text().splitlines()
                # VmRSS counts RAM currently held; VmHWM is the kernel's lifetime peak.
                for field, key in [('VmRSS:', 'process_peak_rss_bytes'), ('VmHWM:', 'process_peak_hwm_bytes')]:
                    value = next(
                        (int(row.split()[1]) * 1024 for row in rows if row.startswith(field)),
                        0,
                    )
                    usage[key] = max(usage[key], value)
            except FileNotFoundError:
                # The original can disappear between reading its PID and /proc.
                pass
            value = int((group / 'memory.current').read_text())
            usage['cgroup_peak_bytes'] = max(usage['cgroup_peak_bytes'], value)
            available = next(
                int(row.split()[1]) * 1024
                for row in Path('/proc/meminfo').read_text().splitlines()
                if row.startswith('MemAvailable:')
            )
            previous = usage['host_min_available_bytes']
            usage['host_min_available_bytes'] = available if previous is None else min(previous, available)
            stop.wait(0.1)  # Unlike sleep(), this returns immediately on shutdown.

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()

    def stop_sampling():
        """Stop the worker before returning its accumulated measurements."""
        stop.set()
        thread.join(timeout=2)
        return usage

    return stop_sampling


def latencies(events, updates, clock_shifts=None):
    """Measure seconds per generation (the completed update being captured).

    Only trainer timestamps need clock_shifts. Skip generations missing the next
    update or filesystem-sync event.
    """
    measurements = []
    for capture in (e for e in events if e['event'] == 'capture_requested'):
        generation = capture['generation']
        # Separate generations so the second restore cannot reuse the first's events.
        phase = {
            event['event']: event['monotonic_ns']
            for event in events if event.get('generation') == generation
        }
        next_update = next((u['completed_ns'] for u in updates if u['update'] == generation + 1), None)
        if next_update is None or 'filesystem_synced' not in phase:
            continue
        shift = (clock_shifts or {}).get(generation, 0)
        next_update += shift

        def seconds(end, start):
            """Reject mixed clock domains instead of publishing negative latency."""
            if end < start:
                raise ValueError('Negative latency: timestamps are not in the same clock domain')
            return (end - start) / 1e9

        # Use each route's completion event; old records fall back to filesystem sync.
        capture_completed = phase.get(
            'dump_completed',
            phase.get('application_save_completed', phase['filesystem_synced']),
        )
        measurements.append({'generation': generation,
            'trainer_to_controller_clock_ns': shift,
            'capture_command_seconds': seconds(capture_completed, phase['capture_requested']),
            'capture_to_sync_seconds': seconds(phase['filesystem_synced'], phase['capture_requested']),
            'restore_to_next_update_seconds': seconds(next_update, phase['restore_requested']),
            'capture_to_original_exit_seconds': seconds(phase['original_exit_verified'], phase['capture_requested']),
            'capture_to_gpu_observed_seconds': seconds(phase['gpu_observed_after_exit'], phase['capture_requested']),
            'post_exit_observation_window_seconds': seconds(phase['gpu_observed_after_exit'], phase['original_exit_verified']),
            'job_b_seconds': seconds(phase['job_b_completed'], phase['job_b_requested']),
            'restore_command_seconds': seconds(phase['restore_returned'], phase['restore_requested'])})
    return measurements


def finish(result, output, controller_offset_ns=None):
    """Attach measurements while retaining raw timestamps and clock offsets.

    A time namespace gives a process an offset view of the host clock. CRIU logs
    the restored offset; old runs need an externally measured controller offset.
    """
    path = output / 'updates.jsonl'
    try:
        updates = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    except json.JSONDecodeError as error:
        # A failed trainer may leave a partial line; keep that failure visible.
        result['measurement_error'] = f'Incomplete update log: {error}'
        return
    result['updates'] = updates
    if 'controller_monotonic_offset_ns' not in result:
        # Preserve a recorded offset when reanalyzing on another process/host.
        if controller_offset_ns is None:
            row = next(s.split() for s in Path('/proc/self/timens_offsets').read_text().splitlines()
                       if s.startswith('monotonic'))
            controller_offset_ns = int(row[1]) * 1_000_000_000 + int(row[2])
        result['controller_monotonic_offset_ns'] = controller_offset_ns
    offsets = {}
    for event in result['events']:
        if event['event'] != 'restore_returned':
            continue
        generation = event['generation']
        offset = result['controller_monotonic_offset_ns']  # Fresh application children inherit it.
        if result['mode'] == 'criu':
            log = (output / f'images-{generation}/restore.log').read_text()
            matches = re.findall(r'timens: monotonic (-?\d+) (\d+)', log)
            if len(matches) != 1:
                raise ValueError('Need exactly one recorded CRIU monotonic clock offset')
            seconds, nanos = map(int, matches[0])
            # Signed seconds + positive nanoseconds: (-25, 700000000) means -24.3 s.
            offset = seconds * 1_000_000_000 + nanos
        offsets[generation] = offset
    result['trainer_monotonic_offsets_ns'] = offsets
    # Both offsets refer to the host clock:
    # controller_time = trainer_time - trainer_offset + controller_offset.
    shifts = {g: result['controller_monotonic_offset_ns'] - offset for g, offset in offsets.items()}
    result['latencies'] = latencies(result['events'], updates, shifts)
    memory = output / 'memory.json'
    if memory.exists():
        # CUDA allocator peaks come from the trainer; RSS was sampled externally.
        result['trainer_memory'] = json.loads(memory.read_text())
