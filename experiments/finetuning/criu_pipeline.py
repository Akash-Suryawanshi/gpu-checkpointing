"""Restore the captured trainer process through CRIU's native CUDA plugin."""

from pipeline import capture_boundary, event, finish, handoff, inspect_restore, launch, session


def run(trial):
    """Capture and restore each requested generation without application state.

    No --save/--load: Python and CUDA state must survive in the process image.
    CRIU's plugin alone owns NVIDIA restoration.
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

        # Reuse the captured PID with fresh image/PID files for each generation.
        trial.pid = session.criu("restore", output, generation, trial.tools, trial.env)
        # Release the saved wait; diagnostics compare before permitting training.
        inspect_restore(trial, generation, expected, request_inspection=True)
    finish(trial)
