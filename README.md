# Deep Learning Homework - Duckietown
BME Deep Learning a gyakorlatban Python és LUA alapon university group assignment <br />
Group name: Tensor.Tensók/T.T <br />
Members: <br />
- Blazsek Péter (AYVNKW) <br />
- Kerekes Márton (UHQUIW) <br />
- Sárdi Ferenc Zsolt (QGGGFC) <br />

Topic: Önvezető autózás a Duckietown környezetben <br />

## Files 
TensorTesok_custom.yaml <br />
- Custom map for the simulations. <br />
<br />

basic_control.py <br />
- Controller for the Duckiebot <br />
<br />

map_perview.mp4 <br />
- Visualization of the simulation on the map. <br />

agents.py <br />
- Used algorimts/agents for our model<br />

traning.py <br />
- For training our model <br />

test.py <br />
- Testing our model <br />

wrappers.py <br />
- Wrappers provided by duckietown_gym with adjustment

utils.py
 - For nicer code
 <br />

## Run the code

Install necessary dependencies by these commands:
```
git clone https://github.com/marton12050/T.T-duckietown
cd T.T-duckietown
pip3 install -e .
```
## Training
- `--seed`  Set seed for training, the default is 123
- `--mode` Set which agent to use `[ppo|impala]`, default is impala
- `--map_name`Set training map, the default is 'udem1'
- `--max_steps` Set maximum step for the training, default is 1000000
- `--num_cpus` Set number of cpu core to use(min 2), default 3
- `--num_gpus` Set number of gpu to use, default 0
- `--training_name` Checkpoint file name
- `--checkpoint` Previous trained modoel checkpoint path to load

Train the model example:
```
python3 training.py --seed 321 --mode ppo --map_name loop_empty
```

## Evaluate
Evaluate on our custom map
- `--seed`  Set seed for training, the default is 123
- `--mode` Set which agent to use `[ppo|impala]`, default is impala
- `--map_name`Set training map, the default is 'TensorTesok_custom'
- `--max_steps` Set maximum step for the training, default is 1000000
- `--checkpoint` Load trained model(checkpoint path)

```
python3 test.py --checkpoint <path_to_checkpoint>
```

## Algorithm
 For training we used the Ray reinforcement learning libary([RLlib](https://docs.ray.io/en/latest/rllib/index.html)) and Ray Tune for optimization and hyperparameter([Ray Tune](https://docs.ray.io/en/latest/tune/index.html)) tuning 

 Used algorithm:

 - [Impala](https://arxiv.org/abs/1802.01561)(Importance Weighted Actor Learner Architecture)
    - Impala is an off-policy actor-critic framework that decouples acting from learning and learns from experience trajectories using V-trace.
 - [PPO](https://arxiv.org/abs/1707.06347)(
 Proximal Policy Optimization) 
    - PPO, is a policy gradient method for reinforcement learning. The motivation was to have an algorithm with the data efficiency and reliable performance of TRPO, while using only first-order optimization. 

## Environment(Wrappers)
Mainly modified from original wrappers are the croppig and reward functions. 

- Rewards
    - Rewards are 0 if the agents do somthing invalid (drive in left lane, not on the road...)
    - Otherwise, reward calculated based on the travel distance, middle of the right lane distane, curve to the middle of right lane, and the speed</br>
    `reward = W1*travel_distance+W2*distance_middle+W3*curve_middle+W4*speed`
- Cropping
    - Crop the sky out(top third of the image) since it has no useful feature

  