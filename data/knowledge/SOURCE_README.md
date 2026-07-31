# Public knowledge resources

Formal experiments require pinned, user-supplied or openly downloaded releases.
Pilot semantic labels are not accepted as substitutes.

## Required layout

```text
data/knowledge/
  loinc/2.82/
    LoincTable/Loinc.csv
    AccessoryFiles/GroupFile/Group.csv
    AccessoryFiles/GroupFile/GroupLoincTerms.csv
    AccessoryFiles/GroupFile/ParentGroup.csv
    LoincLicense_5.8.txt
  hpo/hp.obo
  uberon/uberon.obo
```

Download LOINC 2.82 through a free LOINC account and retain its license file.
Do not commit or redistribute third-party terminology content unless its license
explicitly permits that use.  The formal manifest records versions and SHA-256
checksums.

