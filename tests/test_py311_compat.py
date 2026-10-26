"""The project declares requires-python >= 3.11 (pyproject.toml). PEP 701 (Python 3.12) relaxed f-string syntax, and code written on 3.12+ can silently
stop compiling on 3.11. This test guards every .py file under src/, experiments/ and tests/:

  * on Python < 3.12 it byte-compiles each file (any 3.12-only syntax is a SyntaxError there);
  * on Python >= 3.12 it scans the f-string tokens for the two PEP 701 relaxations that occur in practice:
    a nested string literal that reuses the enclosing f-string's quote, and a backslash inside a replacement field.

It is a guard, not a substitute for running the suite on 3.11 (recommended: a CI job on the lowest supported version).
"""
import glob
import io
import os
import sys
import tokenize

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = sorted(f for d in ("src", "experiments", "tests") for f in glob.glob(os.path.join(ROOT, d, "**", "*.py"), recursive=True))


def _quote(tok_string):
    s = tok_string.lstrip("rRbBuUfF")
    return s[:3] if s[:3] in ('"""', "'''") else s[:1]


def pep701_violations(source):
    """List of (line, reason) for f-string constructs that Python 3.11 rejects. Requires the 3.12+ tokenizer (FSTRING_* tokens)."""
    out, stack, depth = [], [], []                                            # stack of enclosing f-string quotes; brace depth per level
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.FSTRING_START:
            q = _quote(tok.string)
            if stack and depth[-1] > 0 and (q == stack[-1] or (len(stack[-1]) == 1 and q[0] == stack[-1])):
                out.append((tok.start[0], "nested f-string reuses the enclosing quote"))
            stack.append(q)
            depth.append(0)
        elif tok.type == tokenize.FSTRING_END:
            stack.pop()
            depth.pop()
        elif stack and tok.type == tokenize.OP and tok.string in "{}":
            depth[-1] += 1 if tok.string == "{" else -1
        elif stack and depth[-1] > 0:
            if tok.type == tokenize.STRING:
                q = _quote(tok.string)
                if q == stack[-1] or (len(stack[-1]) == 1 and q[0] == stack[-1]):
                    out.append((tok.start[0], "string in a replacement field reuses the enclosing quote"))
                if "\\" in tok.string:
                    out.append((tok.start[0], "backslash inside a replacement field"))
    return out


@pytest.mark.parametrize("path", FILES, ids=lambda p: os.path.relpath(p, ROOT))
def test_source_is_valid_on_python_311(path):
    src = open(path, encoding="utf-8").read()
    if sys.version_info < (3, 12):
        compile(src, path, "exec")
        return
    bad = pep701_violations(src)
    assert not bad, f"{os.path.relpath(path, ROOT)}: Python 3.11 would reject {bad}"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the detector needs the 3.12+ tokenizer")
def test_detector_flags_the_reviewed_construct():
    assert pep701_violations("x = f\"{', '.join(f'{r['k']:.3f}' for r in rs)}\"\n")
    assert pep701_violations("x = f'{d['k']}'\n")
    assert not pep701_violations("x = f\"{d['k']}\"\ny = f\"{', '.join('%.3f' % v for v in vs)}\"\n")
