from treasure_env import TreasureEnv

env = TreasureEnv()
obs, info = env.reset()

action = env.get_action()