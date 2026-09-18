# Secrets Handling

Application secrets live in the managed secret store and are injected at runtime
as environment variables. No secret is stored in a repository, a configuration
file committed to a repository, a ticket, or a chat message.

Every secret has a named owner and a rotation period. Secrets without an owner
are rotated and disabled, since nobody can confirm they are still needed.

Local development uses a separate set of credentials with access only to
development data. Production credentials are never used from a laptop, even
briefly, and even for a read.

A secret exposed in any way is treated as compromised and rotated before any
investigation into how widely it was seen. The investigation happens afterwards,
because rotation is cheap and exposure duration is not recoverable.

Secret scanning runs on every push and blocks the merge on a positive detection.
A developer who needs to commit a string that looks like a secret but is not
must annotate it explicitly rather than disable the scanner.
