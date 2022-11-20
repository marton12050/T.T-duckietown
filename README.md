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
-Custom map for the simulations. <br />
<br />
basic_control.py <br />
-Controller for the Duckiebot <br />
<br />
map_perview.mp4 <br />
-Visualization of the simulation on the map. <br />
<br />
agents.py <br />
-Used algorimts/agents for our model<br />
<br />
traning.py <br />
-For training our model <br />
<br />
test.py <br />
-Testing our model <br />
<br />
wrappers.py <br />
-Wrappers provided by duckietown_gym <br />

## Run the code

Install necessary dependencies by these commands:
```
git clone https://github.com/marton12050/T.T-duckietown
cd T.T-duckietown
pip3 install -e .
```
## Training
- ```--seed```Set seed for training, the default is 123
- ```--mode```Set which agent to use, default is ppo(the only one at the moment)
- ```--map_name```Set training map, the default is 'udem1'
- ```--max_steps``` Set maximum step for the training, default is 100000
- ```--training_name``` Checkpoint file name, default is PPO_training_ckp

Train the model with default parameters:
```
python3 training.py
```

## Evaluate
Evaluate on our custom map
```
python3 test.py
```