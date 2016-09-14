# Project 4: Reinforcement Learning
## Train a Smartcab How to Drive

### Install

This project requires **Python 2.7** with the [pygame](https://www.pygame.org/wiki/GettingStarted
) library installed

### Code

Template code is provided in the `smartcab/agent.py` python file. Additional supporting python code can be found in `smartcab/enviroment.py`, `smartcab/planner.py`, and `smartcab/simulator.py`. Supporting images for the graphical user interface can be found in the `images` folder. While some code has already been implemented to get you started, you will need to implement additional functionality for the `LearningAgent` class in `agent.py` when requested to successfully complete the project. 

### Run

In a terminal or command window, navigate to the top-level project directory `smartcab/` (that contains this README) and run one of the following commands:

```python smartcab/agent.py```  
```python -m smartcab.agent```

This will run the `agent.py` file and execute your agent code.  Here is the sample output:

```
Mean Value Per Step by Varying alpha, gamma
gamma       0.2       0.4       0.6       0.8
alpha
0.2    1.806813  1.840813  1.869663  1.798488
0.4    1.829206  1.881225  1.857167  1.847714
0.6    1.791399  1.841580  1.881225  1.852849
0.8    1.912969  1.791399  1.829206  1.857167


Overall Win Rate by Varying alpha, gamma
gamma    0.2    0.4    0.6    0.8
alpha
0.2    0.990  0.998  0.996  0.988
0.4    0.994  0.988  0.990  0.990
0.6    0.998  0.998  0.988  0.996
0.8    0.998  0.998  0.994  0.990


Mean Penalty by Varying alpha, gamma
gamma    0.2    0.4    0.6    0.8
alpha
0.2   -0.099 -0.103 -0.103 -0.105
0.4   -0.107 -0.120 -0.111 -0.104
0.6   -0.104 -0.121 -0.120 -0.106
0.8   -0.107 -0.104 -0.107 -0.111

State exploration:
Mean  55.3125
Std  2.02233621092


Best in class:
    epsilon 0.0
    win_rate 0.998
    states 51
    score 1.91296928328
    alpha 0.8
    gamma 0.2
    mean penalty -0.107
    stdev penalty 0.427844597956```
