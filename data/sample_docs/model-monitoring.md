# Model Monitoring

Every model in production emits prediction volume, latency percentiles, feature
null rates and an output distribution summary. A model that emits nothing is
treated as failing regardless of whether its predictions look reasonable.

Input drift is measured weekly against the training distribution using a
population stability index per feature. An index above 0.25 on any feature used
by the model raises a drift alert to the owning team.

Drift alerts do not trigger retraining automatically. They open an investigation,
because drift caused by an upstream schema change needs a pipeline fix rather
than a new model, and retraining on corrupted inputs makes the problem
permanent.

Prediction quality is measured against delayed ground truth where it exists,
reported monthly, and compared to the offline evaluation figure recorded at
promotion. A gap larger than ten percent between offline and online performance
is itself a finding worth investigating.

Monitoring dashboards are owned by the team that owns the model, not by the
platform team, who own only the collection pipeline.
