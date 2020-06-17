from dqn.agent import CarRacingDQN
import os
import tensorflow as tf
import gym
import _thread
import re
import sys

model_config = dict(
    min_epsilon=0.1,
    max_negative_rewards=12,
    min_experience_size=int(1e4),
    num_frame_stack=3,
    frame_skip=3,
    train_freq=4,
    batchsize=64,
    epsilon_decay_steps=int(1e5),
    network_update_freq=int(1e3),
    experience_capacity=int(4e4),
    gamma=0.95
)
save_freq_episodes = 400

## Cau hinh
#neu muon training tu dau 
#load_checkpoint = False
#checkpoint_path = "data/checkpoint02"
#train_episodes = float("inf")


# neu muon tiep tuc train tu checkpoint truoc do
load_checkpoint = True
checkpoint_path = "data/checkpoint02"
train_episodes = 0 #chinh so lan train. Dat bang 0 neu muon chay tu nhung ket qua train toi uu nhat

#Setup moi truong game
env_name = "CarRacing-v0"
env = gym.make(env_name)

dqn_agent = CarRacingDQN(env=env, **model_config)
dqn_agent.build_graph()
sess = tf.InteractiveSession()
dqn_agent.session = sess

saver = tf.train.Saver(max_to_keep=100)

#Load checkpoint
if load_checkpoint:
    print("loading the latest checkpoint from %s" % checkpoint_path)
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    assert ckpt, "checkpoint path %s not found" % checkpoint_path
    global_counter = int(re.findall("-(\d+)$", ckpt.model_checkpoint_path)[0])
    saver.restore(sess, ckpt.model_checkpoint_path)
    dqn_agent.global_counter = global_counter
else:
    if checkpoint_path is not None:
        assert not os.path.exists(checkpoint_path), \
            "checkpoint path already exists but load_checkpoint is false"

    tf.global_variables_initializer().run()

#Ham save checkpoint
def save_checkpoint():
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    p = os.path.join(checkpoint_path, "m.ckpt")
    saver.save(sess, p, dqn_agent.global_counter)
    print("saved to %s - %d" % (p, dqn_agent.global_counter))


def play_one_episode():
    reward, frames = dqn_agent.play_episode()
    print("Episode: %d, Total reward: %f, Length: %d, Total steps: %d" %
          (dqn_agent.episode_counter, reward, frames, dqn_agent.global_counter))

    save_cond = (
        dqn_agent.episode_counter % save_freq_episodes == 0
        and checkpoint_path is not None
        and dqn_agent.do_training
    )
    if save_cond:
        save_checkpoint()


def input_thread(list):
    input("...Press Enter to stop after current episode\n")
    list.append("OK")

#tao ham training. lap den khi > train_episodes
def main_loop():
    list = []
    _thread.start_new_thread(input_thread, (list,))
    while True:
        if list:
            break
        if dqn_agent.do_training and dqn_agent.episode_counter > train_episodes:
            break
        play_one_episode()

    print("Done")

#Bat dau train
if train_episodes > 0:
    print("Now training... You can early stop by pressing Enter...")
    print("##########")
    sys.stdout.flush()
    main_loop()
    save_checkpoint()
    print("Training done")

#Bau dau choi
sys.stdout.flush()
dqn_agent.max_neg_rewards = 100
dqn_agent.do_training = False
print("Playing")
print("##########")
sys.stdout.flush()
main_loop()