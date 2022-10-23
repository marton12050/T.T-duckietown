# Deep Learning Homework - Duckietown
BME Deep Learning a gyakorlatban Python és LUA alapon university group assignment <br />
Group name: Tensor.Tensók/T.T <br />
Members: <br />
    - Blazsek Péter (AYVNKW) <br />
    - Kerekes Márton (UHQUIW) <br />
    - Sárdi Ferenc Zsolt (QGGGFC) <br />
Topic: Önvezető autózás a Duckietown környezetben <br />

Duckietown is a platform which connects robotics and AI. The motion of the Duckiebot can be controlled in a simulated 3D enviroment. The goal is appropriate teaching of the controller.

## Files 
TensorTesok_custom.yaml <br />
-Custom map for the simulations. <br />
<br />
basic_control.py <br />
-Controller for the Duckiebot <br />
<br />
map_perview.mp4 <br />
-Visualization of the simulation on the map. <br />
 
## Run the code
To run the code the official gym-duckietown should be installed first (https://github.com/duckietown/gym-duckietown). 

After that to run simulation on the custom map:
```
git clone https://github.com/marton12050/T.T-duckietown
cd T.T-duckietown
./basic_control.py --map-name TensorTesok_custom.yaml
```
