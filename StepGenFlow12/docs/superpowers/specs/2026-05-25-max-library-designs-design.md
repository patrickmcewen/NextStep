# Per-Node Max Library Designs Stopping Criterion

## Summary

Add an optional per-pass `max_library_designs` knob to autotune2. The knob
acts as a per-node cumulative stopping criterion: if a node already has at
least that many successful LLM/library designs, the node is considered
saturated and no new LLM search attempts are launched for it in that pass.

The cap is operational, like `time_limit_seconds`. It limits additional
search work but does not change whether an existing library snapshot is
semantically valid.

## Configuration

`max_library_designs` is accepted anywhere the current pass knobs are accepted:

```yaml
passes:
  - name: tiling
    fewshot: tile_shrink
    time_limit_seconds: 1800
    max_library_designs: 32
  - name: parallel
    fewshot: parallel
    time_limit_seconds: 1800
    max_library_designs: 48
```

Resolution order matches the other pass knobs:

1. per-pass `passes[i].max_library_designs`
2. top-level `max_library_designs`
3. default `null`

`null` means uncapped. Non-null values must be positive integers.

## Counting Semantics

The count is cumulative across passes and resumed state. It counts successful
library designs already admitted for the node, excluding the required
`pass1_baseline` entry.

Prior-pass entries count because they represent successful implementations
already discovered for the node. Resume snapshots usually return before the
criterion is evaluated; if a node has to restart, the prior library still gives
the search enough information to skip saturated nodes.

## Control Flow

```mermaid
flowchart TD
    A[autotune_configs.yaml pass spec] --> B[run_autotune2._resolve_pass_specs]
    B --> C[SearchConfig(max_library_designs)]
    C --> D[autotune._search_node]

    D --> E{matching snapshot exists?}
    E -->|yes| F[return loaded library]
    E -->|no| G[load prior pass library if present]

    G --> H[count cumulative successful designs]
    H --> I{count >= max_library_designs?}

    I -->|yes| J[seed current pass baseline only]
    J --> K[merge prior pass library]
    K --> L[emit variants.py and snapshot]
    L --> M[return capped library]

    I -->|no| N[run search_leaf or search_parent]
    N --> O[admit successful entries]
    O --> P[merge prior pass library]
    P --> Q[emit variants.py and snapshot]
```

## Implementation Points

Update `run_autotune2.py`:

- Add `max_library_designs` to `_PASS_KNOBS`.
- Add default `max_library_designs: None` to `_PASS_SPEC_DEFAULTS`.
- Validate resolved values as `None` or integer `>= 1`.
- Include the resolved value in the pass startup log.
- Pass it into `SearchConfig`.
- Add it to `_STAMP_EXCLUDED_PASS_SPEC_KEYS`, matching
  `time_limit_seconds`, so changing the cap does not invalidate completed
  node snapshots.

Update `src/autotune2/search.py`:

- Add `max_library_designs: int | None = None` to `SearchConfig`.
- Add a helper that counts entries in a `NodeLibrary` excluding entries whose
  provenance is `pass1_baseline`.
- After resume-snapshot loading and after identifying any prior pass library,
  check the cumulative count for the node.
- If the cap is reached, avoid launching LLM attempts. Seed the current
  baseline so the current pass still has a valid library, then merge the prior
  pass library through the existing accumulator path.

## Error Handling

Invalid config should fail during pass-spec resolution with a clear assertion:

- `max_library_designs` must be `null` or a positive integer.
- Floats, strings, zero, and negative values are rejected.

The saturated-node path should write normal node artifacts (`variants.py` and
snapshot when stamping is enabled), so downstream parent nodes and resume logic
continue to see a normal library.

## Testing

Add focused tests in `tests/test_autotune2_multipass.py` and/or
`tests/test_autotune2_search.py`:

- Config resolution accepts top-level and per-pass `max_library_designs`.
- Invalid config values are rejected.
- A node with a prior library count below the cap still invokes search.
- A node with a prior library count at the cap skips LLM attempts.
- The cap excludes `pass1_baseline`.
- Changing `max_library_designs` does not invalidate completed node snapshots.
