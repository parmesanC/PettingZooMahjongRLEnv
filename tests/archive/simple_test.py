import sys
sys.path.insert(0, 'D:\\DATA\\Python_Project\\Code\\PettingZooRLENVMahjong')

from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder

print("Testing Wuhan7P4LObservationBuilder...")

builder = Wuhan7P4LObservationBuilder()
print(f"Builder created: {builder}")
print(f"Builder methods: {[m for m in dir(builder) if not m.startswith('_')]}")

print("\n✓ Basic import and instantiation test passed!")
