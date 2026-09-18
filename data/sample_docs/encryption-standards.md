# Encryption Standards

All data in transit uses TLS 1.2 or above. TLS 1.0 and 1.1 are disabled at the
load balancer and cannot be enabled per service.

Data at rest is encrypted with AES-256. Confidential and restricted datasets use
a customer-managed key rather than a provider-managed one, so that key
destruction is available as a deletion mechanism.

Keys are rotated annually, and immediately on any suspected compromise or on the
departure of anyone who held direct key access. Rotation does not require
re-encrypting historical data where envelope encryption is in use.

Application-level encryption is required for payment credentials and identity
document images, in addition to disk encryption, so that a database dump alone
does not expose them.

Private keys and secrets are never committed to a repository, never passed as
command-line arguments, and never written to application logs. Detection of a
committed key is handled as a severity one incident and the key is rotated
before the commit is removed, because removal alone does not undo exposure.
