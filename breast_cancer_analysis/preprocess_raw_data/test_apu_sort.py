"""Regression for NIAPU's stack overflow when similarities repeat."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("cc"), "requires a C compiler")
class APUSortTests(unittest.TestCase):
    def test_repeated_and_ordered_values(self):
        source = Path(__file__).resolve().parent / "native/apu_label_propagation.c"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Include the actual production sorter; rename the program entry point.
            harness = root / "sort.c"
            harness.write_text(
                '#define main apu_main\n#include "' + str(source) + '"\n'
                '#undef main\n'
                'int main(void) {\n'
                '  int n = 200000; double *x = createVect(n);\n'
                '  for (int i=0; i<n; i++) x[i] = 1;\n'
                '  qSort(x, 0, n-1);\n'
                '  for (int i=0; i<n; i++) if (x[i] != 1) return 1;\n'
                '  for (int i=0; i<n; i++) x[i] = (n-i) % 101;\n'
                '  qSort(x, 0, n-1);\n'
                '  for (int i=1; i<n; i++) if (x[i-1] > x[i]) return 2;\n'
                '  qSort(x, 0, n-1);\n'
                '  for (int i=1; i<n; i++) if (x[i-1] > x[i]) return 3;\n'
                '  free(x); return 0;\n}\n'
            )
            executable = root / "sort"
            subprocess.run(["cc", "-O2", str(harness), "-lm", "-o", str(executable)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
