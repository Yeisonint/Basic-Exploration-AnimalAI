# Basic-Exploration-AnimalAI
Proyecto final del curso Reinforcement Learning - Universidad de los Andes
Autores:
- Alison Gisell Ruiz Ruiz
- Carlos Guillermo Ramirez
- Yeison Estiven Suarez Huertas

# Configuración del entorno
Para evitar problemas con la instalación de python, se recomienda utilizar Anaconda, con instalar solamente Miniconda en el sistema basta. Puede descargar Miniconda desde este [enlace](https://docs.conda.io/en/latest/miniconda.html) y usar la versión con python 3 (cualquier versión).

Si ha instalado Miniconda en su sistema, cree un nuevo entorno para trabajar con este repositorio de la siguiente manera:

```bash
conda create -n final-rl tensorflow-gpu==1.14.0 
conda activate final-rl
```

si su equipo no cuenta con una GPU Nvidia debe eliminar el '-gpu' del comando.

Luego de crear el entorno e ingresar en el, debe instalar los siguientes paquetes:

```bash
pip install -U stable-baselines==2.10.2 gym==0.15.7 gym-unity==0.15.0 animalai==2.0.0 animalai-train==2.0.0 mlagents==0.15.0 mlagents-envs==0.15.0
```

Además de estas librerias, es necesario instalar baselines (OpenAI), descargamos el repositorio e instalamos la libreria con:

```bash
git clone https://github.com/openai/baselines.git
cd baselines
pip install .
```

Con esto ya tenemos la libreria de AnimalAI con sus dependencias.

# Descargar el entorno según el sistema operativo
Tal cual como lo describe en la documentación de AnimalAI, se debe descargar el entorno según el sistema operativo en el cual se esta trabajando y extraer la carpeta AnimalAI en el mismo directorio donde este almacenado este repositorio en su computador:


| OS | Environment link |
| --- | --- |
| Windows | [download v2.0.1](https://drive.google.com/file/d/1BVFAO3pV9DxoPrc6PiDajp2SwCaWZNvJ/view?usp=sharing) |
| Mac | [download v2.0.1](https://drive.google.com/file/d/1dzC3JoDiDhlpVKHXsYi_g-oe9mlIMu6t/view?usp=sharing) |
| Linux |  [download v2.0.1](https://drive.google.com/file/d/18DUEff51hvED5ityNktTpVSaAfZgeKDr/view?usp=sharing) |

Existe una versión 2.0.2 para linux, pero no la vamos a usar. Recuerde que en linux debe darle permisos de ejecución al archivo AnimalAI.

# Ejecutar los entrenamientos
En cada archivo de entrenamiento hay un comentario de ejemplo para la ejecución del algoritmo de aprendizaje, igualmente, para ejecutar cada uno de ellos se tiene que colocar en una consola con el entorno de conda descrito anteriormente.

*Reemplazar los argumentos con valores numericos asociados.

A2C:
```bash
python .\a2c_learn.py total_timesteps alpha gamma epsilon
```

ACER:
```bash
python .\acer_learn.py total_timesteps alpha gamma
```

ACKTR:
```bash
python .\acktr_learn.py total_timesteps alpha gamma
```

DQN:
```bash
python .\dqn_learn.py total_timesteps alpha gamma
```

PPO2:
```bash
python .\ppo2_learn.py total_timesteps alpha gamma
```

# Probar los modelos entrenados
Luego de ejecutar los entrenamientos, los modelos quedan dentro de la carpeta "./models", de ahi se pueden tomar como parámetro para el script *test_train.py*, dentro de la carpeta "./configurations" estan todos los entornos correspondientes al Basic Exploration, asi como en el archivo de configuración *BasicExplorationArenas.yml*.

*El parámetro alg debe ser alguno de estos: a2c,acer,acktr,dqn,ppo2

Test model:
```bash
python .\test_train.py worker_id alg model_path
```