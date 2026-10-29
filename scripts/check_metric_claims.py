"""Cross-check declared document measurements against JSON reports.

A metric line uses `<!-- metric: outputs/reports/file.json#/pointer = 0.123 -->`.
The same line must visibly contain the numeric value. Missing evidence fails.
Unannotated numerical performance table rows fail instead of being accepted.
"""
from pathlib import Path
import argparse
import json
import math
import re
import sys

PATTERN=re.compile(r'<!-- metric: ([^# ]+)#(/[^ ]*) = (-?[\d.eE+]+) -->')


def lookup(report, pointer):
    for key in pointer.lstrip('/').split('/'):
        key=key.replace('~1','/').replace('~0','~')
        report=report[int(key)] if isinstance(report,list) else report[key]
    return report


def check_document(path,root):
    errors=[]
    for lineno,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        for match in PATTERN.finditer(line):
            report,pointer,display=match.groups()
            try:
                actual=float(lookup(json.loads((root/report).read_text(encoding='utf-8')),pointer))
                tolerance=0.5*10**(-len(display.split('.')[1])) if '.' in display and 'e' not in display.lower() else 1e-9
                if not math.isfinite(actual) or abs(actual-float(display))>tolerance:
                    errors.append(f'{path}:{lineno}: {display} differs from {actual}')
                if display not in line[:match.start()]:errors.append(f'{path}:{lineno}: value not visible')
            except (OSError,KeyError,ValueError,TypeError,IndexError) as exc:
                errors.append(f'{path}:{lineno}: missing/invalid metric evidence ({type(exc).__name__})')
        if (line.startswith('|') and re.search(r'\b(?:mIoU|ECE|FPS|latency|precision|recall)\b',line,re.I)
                and re.search(r'(?<![A-Za-z])\d+\.\d+',line)
                and not PATTERN.search(line) and 'NOT RUN' not in line and 'target' not in line.lower()):
            errors.append(f'{path}:{lineno}: numerical performance row has no metric evidence')
    return errors


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    a=ap.parse_args();root=Path(a.root)
    docs=[root/'README.md',root/'MODEL_CARD.md',*sorted((root/'docs').rglob('*.md'))]
    errors=[error for doc in docs for error in check_document(doc,root)]
    for error in errors:print(error)
    print(f'Checked {len(docs)} documents; {len(errors)} evidence errors')
    return int(bool(errors))


if __name__=='__main__':sys.exit(main())
