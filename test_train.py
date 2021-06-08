# Inhabilita las alertas
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_v2_behavior()

# Importa baselines
from baselines import deepq
from baselines.a2c import a2c
from baselines.acer import acer
from baselines.acktr import acktr
import baselines.ppo2.ppo2 as ppo2

# Importa las librerias para el entorno
from animalai.envs.gym.environment import AnimalAIGym
from animalai.envs.arena_config import ArenaConfig

# Lectura de parámetros de entrada
import sys
import os

# Animación y exportar video
import numpy as np
import cv2

def main(worker_id,alg,model_path):
    # Cargar la configuración de la arena
    arenas_configurations = ArenaConfig("test.yml")
    # Crea el entorno
    env = AnimalAIGym(
        environment_filename="AnimalAI/AnimalAI",
        worker_id=worker_id,
        flatten_branched=True,
        uint8_visual=True,
        arenas_configurations=arenas_configurations,
    )
    if alg=='a2c':
        act = a2c.learn(env, network='cnn', total_timesteps=0, load_path=model_path)
    elif alg=='acer':
        act = acer.learn(env, network='cnn', total_timesteps=0, load_path=model_path)
    elif alg=='acktr':
        act = acktr.learn(env, network='cnn', total_timesteps=0, load_path=model_path)
    elif alg=='dqn':
        act = deepq.learn(env, network='cnn', total_timesteps=0, load_path=model_path)
    elif alg=='ppo2':
        act = ppo2.learn(env, network='cnn', total_timesteps=0, load_path=model_path)
    else:
        print('Algoritmo no contemplado')
        exit(-1)
    
    frames=[]
    for _ in range(4):
        obs, done = env.reset(), False
        episode_rew = 0
        tmp_frames = []
        while not done:
            tmp_frames.append(env.render())
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
        if episode_rew>4:
            for frame in tmp_frames:
                frames.append(frame)
    print((frames[0].shape[1], frames[0].shape[0]))
    out = cv2.VideoWriter("./videos/"+alg+".avi",cv2.VideoWriter_fourcc(*'XVID'), 15, (frames[0].shape[1], frames[0].shape[0]))
    for i in range(len(frames)):
        # writing to a image array
        out.write(frames[i])
    out.release()

if __name__ == '__main__':
    argv = [str(arg) for arg in sys.argv[1:]]
    # worker_id, alg(a2c,acer,acktr,dqn,ppo2), model_path

    # python -i .\test_train.py 1 dqn dqn_model.pkl
    main(int(argv[0]),argv[1],argv[2])