import pytest

from kvstreams.envs import build_env


def play_oracle(env):
    start = env.reset()
    assert start.system and start.user
    for _ in range(100):
        step = env.step(env.oracle())
        if step.done:
            return step.reward
    raise AssertionError("episode did not finish")


@pytest.mark.parametrize("name", ["recall", "chainsum"])
def test_oracle_gets_full_reward(name):
    for seed in range(5):
        assert play_oracle(build_env(name, seed=seed)) == 1.0


def test_recall_wrong_answer_scores_zero():
    env = build_env("recall", seed=0, num_distractors=1)
    env.reset()
    env.step("ok")
    env.step("ok")
    assert env.step("nobody").reward == 0.0


def test_same_seed_same_episode():
    a, b = build_env("recall", seed=7), build_env("recall", seed=7)
    assert a.reset() == b.reset()
