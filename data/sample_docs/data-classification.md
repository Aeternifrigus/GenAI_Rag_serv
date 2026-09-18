# Data Classification

Every dataset carries exactly one classification. Where a dataset mixes
classifications, it takes the most restrictive one present until it is split.

Public data may be published without approval. Internal data may be seen by any
employee but not shared outside the company. Confidential data is restricted to
named teams and requires an access request. Restricted data covers client
transaction detail, payment credentials and identity documents, and requires
both an access request and a documented business purpose.

Classification is assigned by the data owner at creation and reviewed annually.
An unclassified dataset is treated as confidential until someone classifies it.

Derived datasets inherit the classification of their most restrictive input.
Aggregation does not lower classification automatically. Lowering the
classification of a derived dataset requires sign-off from the data protection
officer, who will look for re-identification risk.

Classification drives encryption requirements, retention limits and who may
approve an export, but it does not by itself determine how long data is kept.
