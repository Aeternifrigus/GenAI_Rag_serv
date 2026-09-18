# Access Control

Access is granted by role, not by individual. A request for access names the
role being requested and the business reason, and is approved by the role owner
rather than by the requester's manager.

Standing access to restricted data is not granted. Access to restricted data is
time-boxed to a maximum of seven days and expires automatically, with no action
required from anyone to revoke it.

Production database credentials are not issued to people. Human access to
production goes through a brokered session that records the commands run.

Access is reviewed quarterly. A role holder who has not used a role in ninety
days loses it at the next review, and may request it again if the review was
wrong.

Leavers lose all access on their last working day, revoked from the identity
provider rather than service by service. Contractors are given an end date at
account creation, so an unnoticed contract end cannot leave access open.

Shared accounts are prohibited, because an action taken by a shared account
cannot be attributed to anyone.
