# Backup Policy

Production databases are backed up with a full snapshot nightly and continuous
transaction log shipping. The recovery point objective is five minutes and the
recovery time objective is two hours.

Backups are stored in a separate cloud region from the primary, encrypted at
rest with a key held in a different account from the one that can read the
backups. Neither account alone can restore a backup.

Snapshots are kept for thirty-five days. Monthly snapshots are kept for thirteen
months for operational rollback. These periods are backup retention and are
unrelated to the retention obligations that apply to the underlying records.

A restore is rehearsed quarterly against a scratch environment, and the rehearsal
is only counted as passed when an application actually starts against the
restored data. A backup that has never been restored is not evidence of
anything.

Deleting personal data under a subject request also requires purging it from
backups at the next monthly cycle, which is why the restriction mechanism exists
for the interim period.
