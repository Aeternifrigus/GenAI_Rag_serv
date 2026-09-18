# Experiment Tracking

Every training run records its parameters, metrics, code commit, dataset version
and environment. A run that cannot be traced to a commit is not a result and may
not be cited in a promotion decision.

Runs are logged automatically by the training harness. Manual logging is
permitted for exploratory work in notebooks but such runs are marked exploratory
and are excluded from comparison views.

Metrics are recorded on a fixed held-out split that is never used for tuning. A
second split is reserved for the final check before promotion and is read at
most once per candidate model, because a split read repeatedly stops being held
out.

Experiment names follow the pattern of project, objective and date. Nothing
enforces this, but comparison across months becomes impractical without it.

Runs are retained for two years. Artifacts from runs that were never promoted are
deleted after ninety days to control storage cost, while their metrics and
parameters are kept, since the numbers are the part worth keeping.
