import ast
import unittest
from pathlib import Path


EMBC_ROOT = Path(__file__).resolve().parents[3]
TRAINERS = {
    "TruthRL": EMBC_ROOT / "TruthRL/training/verl/verl/trainer/fsdp_sft_trainer.py",
    "CPRO": EMBC_ROOT / "CPRO/verl/trainer/fsdp_sft_trainer.py",
}


def _method(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FSDPSFTTrainer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"missing FSDPSFTTrainer.{name}")


def _loss_scale_default_is_one(func):
    defaults_by_arg = dict(zip(func.args.args[-len(func.args.defaults):], func.args.defaults))
    default = defaults_by_arg.get(next(arg for arg in func.args.args if arg.arg == "loss_scale"))
    return isinstance(default, ast.Constant) and default.value == 1.0


def _is_loss_times_loss_scale(node):
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return False
    names = {side.id for side in (node.left, node.right) if isinstance(side, ast.Name)}
    return names == {"loss", "loss_scale"}


def _is_one_over_n_micro_batches(node):
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and isinstance(node.left, ast.Constant)
        and node.left.value == 1.0
        and isinstance(node.right, ast.Name)
        and node.right.id == "n_micro_batches"
    )


class TestSFTGradAccumScalingStatic(unittest.TestCase):
    def test_sft_backward_loss_is_scaled_for_gradient_accumulation(self):
        for repo, path in TRAINERS.items():
            tree = ast.parse(path.read_text())
            compute_loss = _method(tree, "_compute_loss_and_backward")
            training_step = _method(tree, "training_step")

            self.assertTrue(any(arg.arg == "loss_scale" for arg in compute_loss.args.args), repo)
            self.assertTrue(_loss_scale_default_is_one(compute_loss), repo)

            backward_calls = [
                node
                for node in ast.walk(compute_loss)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "backward"
            ]
            self.assertTrue(any(_is_loss_times_loss_scale(call.func.value) for call in backward_calls), repo)
            self.assertFalse(
                any(isinstance(call.func.value, ast.Name) and call.func.value.id == "loss" for call in backward_calls),
                repo,
            )

            compute_calls = [
                node
                for node in ast.walk(training_step)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_compute_loss_and_backward"
            ]
            self.assertTrue(
                any(
                    keyword.arg == "loss_scale" and _is_one_over_n_micro_batches(keyword.value)
                    for call in compute_calls
                    for keyword in call.keywords
                ),
                repo,
            )


if __name__ == "__main__":
    unittest.main()
