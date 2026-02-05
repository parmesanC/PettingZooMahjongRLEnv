"""Quick test of agent_iter() implementation"""

from example_mahjong_env import WuhanMahjongEnv

print("Creating environment...")
env = WuhanMahjongEnv()

print("Resetting environment...")
obs, info = env.reset()

print(f"Agent selection: {env.agent_selection}")

print("\nTesting agent_iter(num_steps=3)...")
agents = list(env.agent_iter(num_steps=3))
print(f"Agents from iter: {agents}")

print("\nTesting agent_iter() loop...")
count = 0
for agent in env.agent_iter():
    count += 1
    print(f"  Iteration {count}: agent={agent}")
    obs, reward, terminated, truncated, info = env.last()

    if terminated or truncated or count >= 5:
        print("  Stopping test")
        break
    else:
        import random
        action = (random.randint(0, 10), random.randint(0, 34))
        env.step(action)

print(f"\nTest completed! {count} iterations")
