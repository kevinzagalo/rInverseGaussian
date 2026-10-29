"""Keep the published usage examples executable as the API evolves."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import re
import unittest
import warnings


class ReadmeExamplesTests(unittest.TestCase):
    def test_python_examples_execute_in_document_order(self):
        readme = Path(__file__).resolve().parents[1] / "README.md"
        blocks = re.findall(
            r"```python\n(.*?)```", readme.read_text(encoding="utf-8"), re.DOTALL
        )
        self.assertTrue(blocks, "The README should contain executable examples")
        # The KDE example intentionally continues the preceding single-model
        # example, so execute all fences in the same documented namespace.
        namespace = {"__name__": "__readme_examples__"}
        with redirect_stdout(StringIO()), warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for index, block in enumerate(blocks, start=1):
                with self.subTest(example=index):
                    exec(compile(block, f"README.md example {index}", "exec"), namespace)


if __name__ == "__main__":
    unittest.main()
