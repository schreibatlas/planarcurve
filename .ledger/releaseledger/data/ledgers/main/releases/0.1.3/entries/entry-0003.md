---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: 0.1.3
kind: added
summary: Added deterministic trajectory model-order selection and a public heading-rate
  parameterization codec
status: accepted
audience: null
scopes: []
source_refs: []
paths:
- planarcurve/fitting/model_selection.py
- planarcurve/fitting/parameters.py
- planarcurve/fitting/trajectory.py
- planarcurve/fitting/__init__.py
- docs/api/fitting.md
- docs/guides/shared-trajectory-families.md
- examples/select_trajectory_complexity.py
- tests/test_fitting_parameters.py
- tests/test_fitting_trajectory.py
- tests/test_trajectory_model_selection.py
issues: []
prs: []
sources:
- git:66641bc814c0648cc9c84a32aa286b4032256935
contributors:
- '@holgern'
breaking: false
internal: false
order: 3
---
