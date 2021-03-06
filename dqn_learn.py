# Inhabilita las alertas
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_v2_behavior()

# Importa los algoritmos de RL
from baselines import deepq
from baselines import logger

# Importa las librerias para el entorno
from animalai.envs.gym.environment import AnimalAIGym
from animalai.envs.arena_config import ArenaConfig

# Lectura de parámetros de entrada
import sys
import os

# Ejecutar sin GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main(total_timesteps,alpha,gamma):
    # Cargar la configuración de la arena
    arenas_configurations = ArenaConfig("BasicExplorationArenas.yml")
    # Crea el entorno
    env = AnimalAIGym(
        environment_filename="AnimalAI/AnimalAI",
        worker_id=0,
        flatten_branched=True,
        uint8_visual=True,
        arenas_configurations=arenas_configurations,
    )

    # Directorio de registro
    logger.configure(dir="./logs/dqn", format_strs=['stdout', 'log', 'csv', 'tensorboard'])

    # Usando Baselines para entrenar la red
    act = deepq.learn(env=env, network="cnn", total_timesteps=total_timesteps, lr=alpha, gamma=gamma,
        buffer_size=total_timesteps//2,
        exploration_fraction = 0.35,
        print_freq=100,
        train_freq=5,
        learning_starts=total_timesteps//100,
        target_network_update_freq=total_timesteps//400,
        prioritized_replay=True,
        checkpoint_freq=total_timesteps//100,
        checkpoint_path="./logs/dqn",  # Change to save model in a different directory
        dueling=True,
    )

    # Guardar el modelo entrenado
    print("Guardando modelo en dqn_model.pkl")
    act.save("./models/dqn_model.pkl")

if __name__ == "__main__":
    # total_timesteps,alpha,gamma
    argv = [str(arg) for arg in sys.argv[1:]]
    main(int(argv[0]),float(argv[1]),float(argv[2]),int(argv[3]))
    # python -i .\dqn_learn.py 100000 0.001 0.95