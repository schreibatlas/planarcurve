---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.1.1
kind: removed
summary: Removed the legacy `planarcurve.curvature` compatibility module; use `planarcurve.trajectory`
  for heading-rate APIs
status: accepted
audience: null
scopes: []
source_refs:
- git:b16ef50faf7885d2fef795930dc8d5fe8a0cdf90
paths:
- planarcurve/curvature.py
- tests/test_curvature.py
issues: []
prs: []
sources:
- git:b16ef50faf7885d2fef795930dc8d5fe8a0cdf90
contributors:
- '@holgern'
breaking: true
internal: false
order: 1
---
