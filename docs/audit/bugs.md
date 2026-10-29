# Audit bug ledger

Baseline: 03d2dfbc8da7b27be6569494a7df7713dbeeb47b; 232 tracked files. Branch real from main; origin/new already contained. No sim merge.

| ID | Status | File:line | Symptom | Root cause | Fix | Regression test | Commit |
|---|---|---|---|---|---|---|---|
| B001 | fixed | .gitignore:1 | Generated media, inference outputs, camera profiles and partial weights tracked | Incomplete ignore rules and already indexed artifacts | Removed generated artifacts from index; ignore generated trees and models | tests/test_audit_hygiene.py | phase 0 commit |

# Execution limitations

BLOCKED: git pull: fatal: unable to access remote: getaddrinfo() thread failed to start. Remote freshness unverified.

