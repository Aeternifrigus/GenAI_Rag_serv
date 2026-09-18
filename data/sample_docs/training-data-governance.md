# Training Data Governance

Any dataset used to train a model must have a recorded lawful basis and a
recorded classification before training begins. Training on restricted data
requires approval from the data protection officer in addition to the normal
access request.

Personal data used for training is minimised to the fields the model actually
consumes. Fields carried along for convenience are a liability with no benefit.

Training datasets are versioned and immutable. A corrected dataset is a new
version, so that any model can be traced to exactly the rows it saw.

Training data inherits the retention obligation of its source records. Where a
source record must be deleted, the affected dataset version is retired and any
model trained solely on it is scheduled for retraining or retirement.

A model is not itself personal data, but a model that can be made to reproduce
its training examples is treated as though it were, which is why memorisation
checks are run before any model trained on restricted data is promoted.
