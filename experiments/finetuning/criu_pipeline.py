"""Restore the captured trainer process through CRIU's native CUDA plugin."""

from pipeline import capture_boundary, event, finish, handoff, inspect_restore, launch, session


def run(trial):
    """Capture and restore each requested generation without application state.

    The trainer receives neither --save nor --load: its Python objects, threads,
    and CUDA state must survive in the CRIU image. NVIDIA restoration has one
    owner, the CRIU plugin; this pipeline never restores CUDA a second time.
    """
    args = trial.args
    output = args.run_dir
    launch(trial, output, args.capture)
    event(trial, "launched", pid=trial.pid)
    for generation in args.capture:
        expected = capture_boundary(trial, generation)
        session.criu("dump", output, generation, trial.tools, trial.env, trial.pid)
        event(trial, "dump_completed", generation=generation)
        handoff(trial, generation)

        # CRIU reuses the captured numeric PID; session.criu uses fresh image
        # and PID-file paths for each generation, including repeated restores.
        trial.pid = session.criu("restore", output, generation, trial.tools, trial.env)
        # The restored process resumes its existing wait. Allow it to inspect
        # itself first; a separate continue-* marker permits actual training.
        inspect_restore(trial, generation, expected, request_inspection=True)
    finish(trial)
