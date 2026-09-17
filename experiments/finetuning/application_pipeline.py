"""Save explicit training state, then rebuild a fresh trainer from that file."""

from pipeline import capture_boundary, event, finish, handoff, inspect_restore, launch, session


def run(trial):
    """Run the single-capture application-checkpoint comparison in order."""
    args = trial.args
    output = args.run_dir
    checkpoint = output / "application.pt"
    generation = args.capture[0]  # The CLI permits one application capture.
    launch(trial, output, args.capture, ["--save", checkpoint])
    event(trial, "launched", pid=trial.pid)
    expected = capture_boundary(trial, generation)

    # The trainer writes its checkpoint only after this request, so save work is
    # inside the capture measurement. It publishes saved-* and exits normally.
    (output / f"save-{generation}").touch(exist_ok=False)
    session.wait_marker(output / f"saved-{generation}", trial.pid)
    event(trial, "application_save_completed", generation=generation)
    handoff(trial, generation, require_clean_exit=True)

    # This route intentionally reconstructs model/optimizer objects and loads
    # their state. The restarted trainer requests inspection before any update.
    launch(trial, output, extra=["--load", checkpoint])
    inspect_restore(trial, generation, expected)
    finish(trial)
