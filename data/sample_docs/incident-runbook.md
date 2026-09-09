# Production Incident Runbook

An incident is declared when a production service is unavailable, returning errors
above the agreed threshold, or serving materially degraded results to clients.

Severity one covers total unavailability of a client-facing service. Severity two
covers partial degradation affecting a subset of users. Severity three covers
internal-only issues with no client impact.

The on-call engineer acknowledges within 15 minutes for severity one, within one
hour for severity two, and by the next working day for severity three.

Every severity one and severity two incident requires a written postmortem within
five working days. Postmortems are blameless and must identify contributing
factors, not individuals.

Rollback is always preferred over a forward fix during an active severity one
incident. Diagnose after service is restored, not before.
