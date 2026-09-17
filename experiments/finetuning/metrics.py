"""Measure memory and event intervals without adding work to the trainer.

The controller is the running Python program that manages the separate trainer
process. It records events with a monotonic clock: an elapsed-time counter, not
a calendar date. A restored trainer can see a different counter offset, so
finish() aligns timestamps before latencies() calculates intervals. JSON field
names are the evidence API.
"""

import json
from pathlib import Path
import re
import threading


def sample_memory(get_pid):
    """Start a controller-side sampler; return a function that stops and collects it.

    get_pid reads the current trainer PID on each sample because application
    restart creates a new process. Linux identifies a process with an integer
    PID and exposes information about it through virtual files under /proc.
    A cgroup is a group of processes whose resource use Linux measures/limits.
    Its memory includes other processes and cached file data, not just training.
    """
    # This host uses cgroup v2: `0::/path` identifies the controller's group.
    group_path = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]
    group = Path('/sys/fs/cgroup') / group_path.lstrip('/')
    limits = {}
    # Cgroups are nested. A parent's limit still applies when this group says
    # "max" (no additional limit at this level).
    for parent in (group, *group.parents):
        if parent == Path('/sys/fs'):
            break
        if (parent / 'memory.max').exists():
            limits[str(parent)] = (parent / 'memory.max').read_text().strip()
    usage = {'sampling_interval_seconds': 0.1, 'process_peak_rss_bytes': 0,
             'process_peak_hwm_bytes': 0, 'cgroup_peak_bytes': 0,
             'host_min_available_bytes': None, 'cgroup': str(group), 'cgroup_limits': limits}
    # A thread is another execution path within this controller process. The
    # Event lets the main path ask the sampling thread to stop cleanly.
    stop = threading.Event()

    def sample():
        """Collect process, cgroup, and host observations until the stop event."""
        while not stop.is_set():
            try:
                rows = Path(f'/proc/{get_pid()}/status').read_text().splitlines()
                # Resident memory is memory currently held in physical RAM.
                # VmRSS is current usage; VmHWM is the Linux kernel's peak.
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
    """Calculate one set of intervals per capture generation, in seconds.

    A generation is the completed update being captured (2 or 3). Controller
    events already share a clock; only trainer update times need clock_shifts.
    Failed or incomplete trials may lack an endpoint and yield no interval.
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

        # CRIU finishes a dump; the application route finishes its own save.
        # Sync is a fallback for older event records without a command endpoint.
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
    """Attach measurements to a run result, retaining the raw timing evidence.

    A Linux namespace gives processes their own view of a resource. A time
    namespace can shift the clock they see without changing the host's clock.
    Old runs need an externally measured controller offset. New runs read their
    own namespace. CRIU's restore log supplies the restored trainer's offset;
    fresh application children inherit the controller's offset instead.
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
            # Linux represents negative fractions as signed seconds + positive ns.
            # For example, (-25, 700000000) means -24.3 seconds, not -25.7.
            offset = seconds * 1_000_000_000 + nanos
        offsets[generation] = offset
    result['trainer_monotonic_offsets_ns'] = offsets
    # Both offsets are measured against the underlying host clock:
    # controller_time = trainer_time - trainer_offset + controller_offset.
    shifts = {g: result['controller_monotonic_offset_ns'] - offset for g, offset in offsets.items()}
    result['latencies'] = latencies(result['events'], updates, shifts)
    memory = output / 'memory.json'
    if memory.exists():
        # CUDA allocator peaks come from the trainer; RSS was sampled externally.
        result['trainer_memory'] = json.loads(memory.read_text())
