# Shared POOL-EN inputs

One copy of the source data for all conditions. Do not duplicate these files under `condition_*`.

```text
data/
├── splits/                 # reddit jsonl
│   ├── user_*_query.jsonl       # 500 query profiles (500 comments each)
│   └── user_*_candidate.jsonl   # 1000 candidate profiles (500 comments each)
└── profiles/
    └── profiles_query.jsonl     # same 500 query users, Staab-format (Condition C)
```

| Who reads it | Path |
|--------------|------|
| A, B, So1, So2 | `splits/` |
| C | `profiles/profiles_query.jsonl` |
