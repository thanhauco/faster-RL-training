"""Count the work saved by compacting in place instead of re-prefilling."""

from kvstreams.benchmark import format_table, run_benchmark

results = run_benchmark(budgets=[256, 512], num_rollouts=2, policy="markovian", policy_kwargs={"keep_last": 1})
print(format_table(results))
