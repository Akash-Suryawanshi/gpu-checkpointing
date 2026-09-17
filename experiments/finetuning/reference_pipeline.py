"""Establish reproducibility with two independently initialized training runs."""

from pipeline import event, finish, launch


def run(trial):
    """Finish two independent runs; run.py compares them before admitting reuse."""
    output = trial.args.run_dir
    launch(trial, output)
    event(trial, "launched", pid=trial.pid)
    finish(trial)

    # A new Python process must recreate the same state without a saved trainer.
    # Its own directory keeps markers, losses, and fingerprints independent.
    repeat = output / "repeat"
    repeat.mkdir()
    launch(trial, repeat)
    finish(trial, record_exit=False)
