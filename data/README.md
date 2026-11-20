# Dataset acquisition

Use the **Churn Modelling** listing by `shrutimechlearn`:

[Original Kaggle source and download page](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling).

Download through the publisher's supported interface, review the current terms, and place `Churn_Modelling.csv` here. Do not use an employer/customer export or silently substitute a newer synthetic churn competition dataset.

The local original contains 10,000 records. Required predictors are:

```text
CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
```

The binary target is `Exited`. Common exports additionally contain `RowNumber` or `#`, `CustomerId`, and `Surname`. These identifiers are deliberately excluded; the exact original local export uses `#`. The code validates named predictors rather than relying on column positions.

`SOURCE_PROVENANCE.json` records the original local file's hash, count, and schema. A download with different row ordering, identifiers, newline conventions or data version can have a different byte hash. The result manifest always records the actual input used; report new results rather than pretending an unmatched file reproduced an old run.

The listing was checked on 20 November 2026. Its dynamically served licensing text was not independently readable through the audit tools, so this repository asserts no specific data license. Raw CSVs are excluded from Git, and the code license does not grant data redistribution rights. The source-only workflow remains reproducible through the publisher's legitimate download route.
