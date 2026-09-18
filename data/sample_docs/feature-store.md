# Feature Store

Features are defined once and served to both training and inference from the
same definition, which is the mechanism that prevents training and serving skew.
A feature computed separately in a training script is not a feature, it is a
local variable.

Each feature declares an owner, a freshness expectation and a backfill window.
Consumers see the freshness of every feature they read, so a stale feature
degrades a prediction visibly rather than silently.

Point-in-time correctness is enforced on training reads. A training set assembled
without it will leak future information into the past and produce an offline
score that cannot be reproduced online.

Features are versioned. Changing a feature's computation creates a new version
rather than mutating the existing one, because models already in production are
reading the old one.

Deprecating a feature version requires notifying every registered consumer and a
minimum notice period of one month.
