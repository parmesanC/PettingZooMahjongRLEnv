# Fix PettingZoo AECEnv `_cumulative_rewards` Attribute

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the `AttributeError: 'WuhanMahjongEnv' object has no attribute '_cumulative_rewards'` error by properly implementing PettingZoo AECEnv's required reward tracking mechanism.

**Architecture:** The `WuhanMahjongEnv` class inherits from PettingZoo's `AECEnv`, which requires proper initialization of `_cumulative_rewards` dict in `reset()` and proper reward accumulation in `step()`.

**Tech Stack:** PettingZoo AECEnv, Python 3.x

---

## Context: Understanding the Error

The error occurs because PettingZoo's `AECEnv.last()` method (called by the manual controller) requires `_cumulative_rewards` attribute to be initialized. This attribute tracks cumulative rewards for each agent across steps.

According to [PettingZoo Environment Creation documentation](https://pettingzoo.farama.org/content/environment_creation/), the `reset()` method must initialize:
- `agents`
- `rewards`
- `_cumulative_rewards` ← **MISSING**
- `terminations`
- `truncations`
- `infos`
- `agent_selection`

---

## Task 1: Add `_cumulative_rewards` Initialization in `reset()`

**Files:**
- Modify: `example_mahjong_env.py:102-125`

**Step 1: Read the current reset() method**

The `reset()` method currently initializes agents, rewards, terminations, truncations, and infos, but is missing `_cumulative_rewards`.

**Step 2: Add _cumulative_rewards initialization**

After line 124 (`self.infos = {agent: {} for agent in self.possible_agents}`), add:

```python
# Initialize cumulative rewards for PettingZoo AECEnv compatibility
self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
```

**Step 3: Verify the change**

Run: `python -c "from example_mahjong_env import WuhanMahjongEnv; env = WuhanMahjongEnv(); env.reset(); print('_cumulative_rewards:', env._cumulative_rewards)"`

Expected output: `_cumulative_rewards: {'player_0': 0.0, 'player_1': 0.0, 'player_2': 0.0, 'player_3': 0.0}`

**Step 4: Run the game to verify the fix**

Run: `python play_mahjong.py --mode human_vs_ai --renderer cli`

Expected: The AttributeError should be resolved. Game should start properly.

**Step 5: Commit**

```bash
git add example_mahjong_env.py
git commit -m "fix(env): add _cumulative_rewards initialization for PettingZoo AECEnv compatibility"
```

---

## Task 2: Add Reward Accumulation in `step()`

**Files:**
- Modify: `example_mahjong_env.py:157-224`

**Step 1: Understand reward accumulation**

PettingZoo's `AECEnv` provides an inherited `_accumulate_rewards()` method that adds current `self.rewards` to `self._cumulative_rewards` for each agent. This must be called at the end of each `step()`.

**Step 2: Add _accumulate_rewards() call**

At the end of the `step()` method (before the return statement on line 224), after line 218 (`info = self._get_info(agent_idx)`), add:

```python
# Accumulate rewards for PettingZoo AECEnv compatibility
self._accumulate_rewards()
```

**Step 3: Verify reward accumulation works**

Run: `python play_mahjong.py --mode human_vs_ai --renderer cli`

Play a few turns and verify no errors occur.

**Step 4: Commit**

```bash
git add example_mahjong_env.py
git commit -m "fix(env): add reward accumulation for PettingZoo AECEnv compatibility"
```

---

## Task 3: Reset Cumulative Rewards for Current Agent at Step Start

**Files:**
- Modify: `example_mahjong_env.py:157-224`

**Step 1: Understand the reward reset pattern**

In PettingZoo AECEnv, when `last()` is called for an agent, it returns that agent's `_cumulative_rewards`. Then, at the start of the next `step()`, that agent's `_cumulative_rewards` should be reset to 0 (since the rewards were already "consumed" by `last()`).

**Step 2: Add cumulative rewards reset for current agent**

At the beginning of the `step()` method (after line 167: `current_agent = self.agent_selection`), add:

```python
# Reset cumulative rewards for the current agent (rewards were already returned by last())
if current_agent in self._cumulative_rewards:
    self._cumulative_rewards[current_agent] = 0.0
```

**Step 3: Verify the fix**

Run: `python play_mahjong.py --mode human_vs_ai --renderer cli`

Play through a complete game and verify that:
- No AttributeError occurs
- Rewards are properly tracked
- The game can complete multiple rounds

**Step 4: Commit**

```bash
git add example_mahjong_env.py
git commit -m "fix(env): reset cumulative rewards for current agent at step start"
```

---

## Verification Steps

After completing all tasks:

**Step 1: Run the manual control test**

```bash
python play_mahjong.py --mode human_vs_ai --renderer cli
```

**Expected results:**
- Game starts without AttributeError
- Can play through multiple rounds
- Special tiles (赖子/皮子) are displayed correctly
- Discard pile is empty at the start of each round

**Step 2: Test with API test**

```bash
python -c "
from example_mahjong_env import WuhanMahjongEnv
env = WuhanMahjongEnv()
obs, info = env.reset()
print('Reset successful')
print('Agents:', env.agents)
print('Cumulative rewards:', env._cumulative_rewards)

# Simulate a few steps
for i in range(3):
    obs, reward, terminated, truncated, info = env.last()
    print(f'Step {i}: reward={reward}')
    if terminated or truncated:
        break
    env.step((0, 0))  # Dummy action
"
```

**Step 3: Commit final verification**

```bash
git add docs/plans/2026-01-21-cumulative-rewards-fix.md
git commit -m "docs: add cumulative rewards fix implementation plan"
```

---

## Summary

This fix implements the three required components for PettingZoo AECEnv reward tracking:

1. **Initialization** (`reset()`): Create `_cumulative_rewards` dict with all agents set to 0
2. **Accumulation** (`step()` end): Call `_accumulate_rewards()` to add current rewards to cumulative
3. **Reset** (`step()` start): Reset current agent's cumulative rewards to 0 after they were returned by `last()`

These changes ensure full compatibility with PettingZoo's AEC API and resolve the AttributeError.

---

**Sources:**
- [PettingZoo Environment Creation](https://pettingzoo.farama.org/content/environment_creation/)
