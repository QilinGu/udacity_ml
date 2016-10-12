# Self-driving Cars with Tensorflow
## A Capstone Project

This project was submitted in partial fulfillment
of the Udacity Machine Learning Engineer Nanodegree.

Scott Penberthy  
November 1, 2016

_[Ed. This is a work in progress, publishing by the end of October if I'm lucky!]_

## I. Definition

### Project Overview
Self-driving cars are fascinating.  However, the learning curve is steep.  We noticed a lack of simple environments
for experimenting with the underlying algorithms in pedagogical settings.  We have built one such environment here, which a programmer can run just fine _without_ expensive GPUs.  

Our virtual environment is a derivative of 
[Matt Harvey's virtual car](https://medium.com/@harvitronix/using-reinforcement-learning-in-python-to-teach-a-virtual-car-to-avoid-obstacles-6e782cc7d4c6#.58wi2s7ct), 
ported to work with TensorFlow, Python 2.7, and PyGame 5.0. The
machine learning algorithm is based on 
[Deep Q Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf), 
without the "deep" part.  Many of the ideas
were derived from [songotrek's Tensorflow implementation](https://github.com/songrotek/DQN-Atari-Tensorflow/blob/master/BrainDQN_Nature.py) of DeepMind's
[Atari-playing Deep Q Learner](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)


### Problem Statement
![fig1](https://cdn-images-1.medium.com/max/1200/1*K11fcwyorgnbTcl5dEnxVw.jpeg)
Your job is to write an AI for a simulated toy car that learns to drive by itself.  

The figure above captures our toy car scenario.  The car is shown in green, sensing the environment
via three sonar sensors. Three slow-moving obstacles are shown in blue.  A cat darts around the
environment in orange.   Our challenge is to build a learning algorithm that learns to drive
without hitting things.  The car's throttle is stuck in the "on" position.  Hey, its a cheap toy.

### Prerequisites
1. [Anaconda Python Distribution, 2.7](https://www.continuum.io/why-anaconda) for Python
2. [Tensorflow for Anaconda](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#anaconda-installation) for AI
3. [PyGame](http://www.pygame.org/wiki/GettingStarted) for graphics
4. [PyMunk](http://www.pymunk.org/en/latest/) for physics
5. [Numpy](http://www.numpy.org/) for math

This has been successfully tested on a MacBook Pro running OS X El Capitan.

### Running the code
After successfully cloning this repository and installing all the prerequisites, ```cd``` to
the repository directory so we have access to ```learning.py``` and ```carmunk.py```.  Let's activate
the TensorFlow environment and launch an interactive python shell:

```
% source activate tensorflow
(tensorflow) % ipython
Python 2.7.12 |Continuum Analytics, Inc.| (default, Jul  2 2016, 17:43:17) 
Type "copyright", "credits" or "license" for more information.

IPython 5.1.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: 
```

Next, let's import our file then create our Deep Q learner environment in 
one line of code!
```python
In [1]: from learning import *

In [2]: ai = Learner()
```

_[ed. At the time of this writing, you may see warning statements on a Mac about using outdated audio controls.  That's OK.  We don't
need audio for this task.]_

The ```ai``` object is our Deep Q Learner, without the deep, controlling a simluated game ```g```
analyzed by a Tensorflow session ```s```.  Let's get into it.  Try the following to demo 1000 steps of
the simulator and watch the car drive!  At this point the algorithm is purely random.

```python
In [3]: ai.demo()
```
You can try each step yourself, going straight, left or right.  You'll see three items returned,
the reward, an array of sensor readings, and a boolean indicating whether you've hit a wall
or object:

```python
In [4]: ai.g.step(0)
(-2.0, array([ 6.        ,  4.        ,  6.        ,  0.13147931,  0.16259441,
         4.68615124]), False)

In [5]: ai.g.step(1)
(-2.0, array([ 4.        ,  4.        ,  6.        ,  0.12923619,  0.14867274,
         4.48615124]), False)


In [6]: ai.g.step(2)
(-3.0, array([ 5.        ,  3.        ,  5.        ,  0.12897384,  0.13439194,
         4.68615124]), False)
```


### Metrics
We will evaluate our algorithm by comparing its top score performance in
1000 games against an agent that picks actions randomly.  That's all that 
matters in the end -- bragging rights for the most number of iterations
before a crash.  We need to do (much) better than random chance.

We will track learning progress with "Q Max" from the Deep Q Learning paper. Qmax is essentially
a measure of agent confidence.  As the agent learns, its better able to predict what action to take, leading to
higher rewards.  The "Q" value tells the agent how much it believes a given state is worth in terms of 
longterm reward.  The longer the agent stays alive, the greater the reward, the higher QMax. Technically, Qmax 
is the maximum value seen by the neural network that's learning to estimate the Q function,
across all training examples, across time.

We'll use the TensorBoard visualizer available with Tensorflow to see our top score and QMax change over time.
Our learning agent sits and observes for several thousand iterations before learning.  Like Mom taught us, 
it pays to listen and observe before making judgment!

Tensorboard runs as a local Java application, serving web pages as its GUI on a local port.  This is Google, after all.
The visualizer reads log data from the "train" subdirectory and periodically updates the display as log data
grows.  I find that periodically cleaning this directory is quite useful.

To see this work, let's mute the graphic display and run 25,000 iterations:
```python
In [7]: ai.mute()
True

In [8]: for i in range(25000): ai.step()
```
Now launch another terminal and ```cd``` to the repository directory, activating tensorflow
as before.  You'll see new content in the
log directory.  Your log filename will be different than mine, a combo of time and your local machine
name.  Launch tensorboard and point to this directory.
```
% source activate tensorflow

(tensorflow) % ls train
events.out.tfevents.1476194157.Scotts-MBP.home 

(tensorflow) % tensorboard --logdir train
Starting TensorBoard 23 on port 6006
(You can navigate to http://0.0.0.0:6006)
```
Launch your browser and navigate to http://0.0.00:6006.  You'll see three numbers we're tracking, loss, qmax and
score. Loss represents the amount of error in a random sample of historical frames, taken every learning
cycle.  QMax and Score are tracked over time, too.  Click to reveal or hide plots.

If you're curious, click on the "histograms" tab to see our network weights and biases change over time.
Here, each slice in time (vertically) is a modified box plot, which
shows the first and second standard deviations as a band of dark (1) and lighter (2) orange,
with a faded orange for outliers (beyond 2 standard deviations).  When we plot these bands closely together
and connect the regions, we get a flowing orange shape showing our distribution "on the side" as the
means shifts over time.

The separate terminal now shows the live log data of the
Tensorboard web server.  I often let this terminal sit idle to 
monitor activity in a separate window.  When I try a new model, I often stop the application with control-c, then
eliminate log files from the ```train`` directory, and restart.  Crude, yes.  Effective?  You bet.

## II. Analysis

### Data Exploration
The simulator provides the following information about the car at every cycle:
- s1, s2, s3 - 3 sensor readings from 0-40
- x - x position, [0,1] where 0 is far left, 1 far right
- y - y position, [0,1] where 0 is top, 1 bottom
- theta - the heading angle of the car, 0 to 2*pi

You're allowed to take three actions:
- 0, stay on course
- 1, turn left by 0.2 radians
- 2, turn right by 0.2 radians

In addition to the sensor readings, the simulator also returns the following at
each iteration:
- Reward, an integer in [-100, 10] where negative values are bad, positive values are good
- Terminal, a boolean indicating whether a crash has occurred

Our challenge is to choose an action (0,1 or 2) at each time step.  We're only given the state (s1,s2,s3,x,y,theta), 
a prior reward, and a boolean.  

When we're driving free and clear, the reward varies over
the interval [-4, 34] which represents the shortest distance record by a sonar sensor offset by -6.  Thus, if one
sensor has a reading of 2, the reward will be -4.  

A crash occurs when a sensor reading of 1 senses an object, returning a reward of -100.  The simulator randomly shifts
and rotates the car in an attempt to "get free" from the obstacle.

The x and y position are floating point numbers varying from 0 to 1, indicating how far along each axis we sit.  The
angle is measured in radians, varying from 0 to 2*Pi.  These measurements are a replacement for "SLAM" technology
that simultaneously creates maps of the environment from richer sensor data.  The hope here is that the learning
algorithm figures out to stay closer to the middle, to turn away from walls, and to avoid objects when they're 
getting close.

### Exploratory Visualization
![PyGame Racer v1](http://i.makeagif.com/media/10-12-2016/BksdP7.gif)
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
