# Multi-Agent-Interaction-Reinforcement-Learning
Applying reinforcement learning techniques to multiple agents in a custom environment

![](images/landing.gif)

This projects uses reinforcement learning to try and simulate a competitive Capture-the-Flag game played between two teams in the environment with static obstacles (walls) present.

To run the program, first clone this repository into your local folder.

Make sure that Python is installed on your system.

To install external packages, from the project's root directory run:

```pip install -r requirements.txt```

How to run the application: from the root directory run:  
    ``` python game.py ```  
    Then follow the instructions on the GUI.


If you have trouble running the application:

Possible Problems:
1. The script expects your python alias to be ```python3 file.py```
    if it is not this change lines 9 and 12 in game.py.
    e.g. ```subprocess.call(["python"... ```
