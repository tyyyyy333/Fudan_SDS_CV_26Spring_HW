# Scripts

```text
entrypoints/
  setup/    environment bootstrap and verification
  task1/    official Task 1 training, preview, and final rendering commands
  task2/    official ACT training and rollout commands
tools/      shell utilities that are not primary experiment entrypoints
archive/    superseded experiments and legacy fusion implementations
*.py        reusable data, rendering, evaluation, and report implementations
```

Use shell commands from `entrypoints/` for reproducible runs. Python files in
this directory expose the implementation and usually provide `--help`.
Archived scripts must not be used to generate submission results.
