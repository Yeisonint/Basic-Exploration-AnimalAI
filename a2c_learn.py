# # # Inhabilita las alertas
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_v2_behavior()

# Importa AnimalAI
from animalai.envs.gym.environment import AnimalAIGym
from animalai.envs.arena_config import ArenaConfig
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

# Importa Baselines
from baselines.bench import Monitor
from baselines import logger
from baselines.a2c import a2c

# Otras librerias
import sys
import os

def make_aai_env(env_directory, num_env, arenas_configurations, start_index=0):
    def make_env(rank, arena_configuration):
        def _thunk():
            env = AnimalAIGym(
                environment_filename=env_directory,
                worker_id=rank,
                flatten_branched=True,
                arenas_configurations=arena_configuration,
                uint8_visual=True,
            )
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env

        return _thunk
    return SubprocVecEnv(
        [make_env(i + 20 + start_index, arenas_configurations) for i in range(num_env)]
    )


def main(total_timesteps,alpha,gamma,epsilon):
    arenas_configurations = ArenaConfig("BasicExplorationArenas.yml")
    
    # Directorio de registro
    logger.configure(dir="./logs/a2c", format_strs=['stdout', 'log', 'csv', 'tensorboard'])

    env = make_aai_env("AnimalAI/AnimalAI", 1, arenas_configurations)
    act = a2c.learn(env=env, network="cnn", total_timesteps=total_timesteps, lr=alpha, gamma=gamma,epsilon=epsilon)

    # Guardar el modelo entrenado
    print("Guardando modelo en a2c_model.pkl")
    act.save("./logs/a2c/a2c_model.pkl")

if __name__ == "__main__":
    # total_timesteps,alpha,gamma
    argv = [str(arg) for arg in sys.argv[1:]]
    main(int(argv[0]),float(argv[1]),float(argv[2]),float(argv[3]))
    # python -i .\a2c_learn.py 100000 0.001 0.95 0.0001