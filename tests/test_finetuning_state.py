"""State continuity and corruption checks using a small real Adam/dropout model."""

import copy
from pathlib import Path
import random
import tempfile
import unittest

import torch
import state


def trainer(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.1), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: 1 - n / 4)
    return model, optimizer, scheduler


def update(model, optimizer, scheduler, progress):
    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    progress["update"] += 1
    progress["next_example_index"] += 1
    return loss.item()


class StateTests(unittest.TestCase):
    def test_application_roundtrip_preserves_the_next_stochastic_update(self):
        torch.set_num_threads(1)
        model, optimizer, scheduler = trainer(7)
        progress = {"update": 0, "next_example_index": 0}
        update(model, optimizer, scheduler, progress)
        before = state.inspect(model, optimizer, scheduler, progress, {}, {})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            state.save_application(path, model, optimizer, scheduler, progress, {}, {})
            expected_loss = update(model, optimizer, scheduler, progress)
            expected = state.inspect(model, optimizer, scheduler, progress, {}, {})
            restored, restored_opt, restored_sched = trainer(99)
            resumed = state.load_application(path, restored, restored_opt, restored_sched, {}, {})
            state.compare(before, state.inspect(restored, restored_opt, restored_sched, resumed, {}, {}))
            self.assertEqual(expected_loss, update(restored, restored_opt, restored_sched, resumed))
            state.compare(expected, state.inspect(restored, restored_opt, restored_sched, resumed, {}, {}))

    def test_corruption_is_identified_even_when_adapter_weights_match(self):
        model, optimizer, scheduler = trainer(7)
        progress = {"update": 0, "next_example_index": 0}
        update(model, optimizer, scheduler, progress)
        record = state.inspect(model, optimizer, scheduler, progress, {}, {})
        paths = [("optimizer", "states", "0.weight", "exp_avg", "sha256"),
                 ("progress", "next_example_index"), ("rng", "cpu", "sha256"), ("schedule", "last_epoch")]
        for path in paths:
            with self.subTest(field=path):
                corrupted = copy.deepcopy(record)
                leaf = corrupted
                for key in path[:-1]:
                    leaf = leaf[key]
                old = leaf[path[-1]]
                leaf[path[-1]] = "changed" if isinstance(old, str) else old + 1
                self.assertEqual(record["adapters"], corrupted["adapters"])
                with self.assertRaisesRegex(ValueError, path[-1]):
                    state.compare(record, corrupted)


if __name__ == "__main__":
    unittest.main()
