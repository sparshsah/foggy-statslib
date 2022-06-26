Closed issue status can be `DONE`, `WONTDO`, or `WORKSFORME`.

We triage open accepted issues into the following priority levels:
* `DEFCON_0`: Chosen security scheme is fatally flawed (e.g. Classical RSA after quantum supremacy).
* `DEFCON_1`: Chosen security scheme is theoretically sound, but implementation has a fundamental bug.
* `DEFCON_2`: Chosen security scheme is theoretically sound and implementation is essentially correct,
              but breaks in presence of a practical corner case (e.g. race condition).
* `DEFCON_3`: Functionality has a critical fundamental logic error (e.g. sign flip).
* `DEFCON_4`: Functionality has a less-critical fundamental logic error (e.g. off-by-one).
* `DEFCON_4`: Functionality breaks in presence of a practical corner case (e.g. negative int).
* `DEFCON_5`: Functionality breaks API's promise, but delivers a not-unreasonable alternative
              (e.g. promises geometric returns, but delivers logarithmic returns).
* `DEFCON_6`: BAU enhancement.
* `DEFCON_7`: Long-term enhancement.

Before acceptance, every issue is `NEW`.
