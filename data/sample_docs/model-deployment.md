# Model Deployment Standard

Every model promoted to production must have a recorded training run including
the dataset version, hyperparameters, evaluation metrics and the resulting
artifact hash. Runs that cannot be reproduced are not eligible for promotion.

Models are served behind a versioned endpoint. The previous version remains
available for at least 14 days after a new release so traffic can be rolled back
without a redeploy.

Feature transformations applied at training time must be applied identically at
inference time. Training and serving code must share the same transformation
module rather than reimplementing it.

Every production model has a monitored input distribution. A significant shift in
input distribution raises an alert but does not automatically trigger retraining,
because a shift may reflect a genuine change in the world or an upstream data
fault, and those require different responses.

Retraining is approved by a human after reviewing the drift evidence.
