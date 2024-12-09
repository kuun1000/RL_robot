from XarmRobot import XarmRobotEnv
import pybullet as p

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

def create_train_env():
    env = XarmRobotEnv(is_for_training=True)
    return Monitor(env)  # Avvolgi l'ambiente con Monitor

def create_test_env():
    env = XarmRobotEnv(is_for_training=True)
    return Monitor(env)  # Avvolgi l'ambiente con Monitor


if __name__ == '__main__':

    # Definisce la cartella per tensorboard
    log_dir = "./tensorboard_logs/"

    # Numero di ambienti in parallelo
    num_envs = 4

    # Parallelizzazione dell'ambiente di addestramento
    train_env = SubprocVecEnv([create_train_env for _ in range(num_envs)])
    print("Training environment initialized and reset successfully")
    train_env.reset()

    # Definizione del modello DQN con parametri ottimizzati
    model = DQN(
        'CnnPolicy',
        train_env,
        learning_rate=5e-4,
        buffer_size=200000,
        learning_starts=7500,           # Aumentato per dare più tempo all'esplorazione iniziale
        batch_size=256,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        train_freq=4,                   # Da rivedere
        target_update_interval=5000,
        verbose=1,
        tensorboard_log=log_dir  # Logging per TensorBoard
    )

    # Configurazione delle callback per i checkpoint e la valutazione
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path='./checkpoints/',
        name_prefix='dqn_model'
    )

    eval_callback = EvalCallback(
        DummyVecEnv([create_test_env]),
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=1000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    # Lista dei callback da passare al modello
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Addestramento del modello con i parametri ottimizzati
    model.learn(total_timesteps=40_000, callback=callback)  # Aumentato il numero di timestep
    model.save("dqn_Z_6")

    # Disconnette PyBullet dopo l'addestramento
    if p.isConnected():
        p.disconnect()

    # Chiudi l'ambiente di addestramento
    train_env.close()
