# Metareasoning.jl

Source code for the paper [Tuning the Hyperparameters of Anytime Planning: A Metareasoning Approach with Deep Reinforcement Learning](https://abhinavbhatia.me/publication/BSNZicaps22)
(Bhatia, A., Svegliato, J., Nashed, S. B., & Zilberstein, S. (2022). In _Proceedings of the International Conference on Automated Planning and Scheduling, 32_(1), 556-564. https://ojs.aaai.org/index.php/ICAPS/article/view/19842)


This package provides RL environments (compatible with [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl) API) for controlling hyperparameters of anytime algorithms [RRTStar.jl](https://github.com/bhatiaabhinav/RRTStar.jl) and [AnytimeWeightedAStar.jl](https://github.com/bhatiaabhinav/AnytimeWeightedAStar.jl).

## Install Instructions

1. Clone this repository
2. Install [Julia](https://julialang.org/) 1.7.2.
3. Within the root directory of this package, run the following command:
```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

You will need a [wandb](https://wandb.ai/site) account to log the runs. When you run either `rrt_dqn.jl` or `aastar_dqn.jl` for the first time, you will be asked to login to your wandb account.

---------------------------------------------------------

## Training and evaluating models

### RRT*

Go through the file `rrt_dqn.jl`, and run:

```bash
julia --project=. rrt_dqn.jl
```

This will train a dqn model, evaluate it, and record videos of sample episodes. Look for the logs in `logs/` directory and on your wandb dashboard.

<center><img src="https://github.com/bhatiaabhinav/Metareasoning.jl/blob/main/rrt_example.gif" width="300"/></center>


### Anytime Weighted A* (AWA*)

Go through the file `aastar_dqn.jl`, uncomment the code specific to the desired search problem, and run:

```bash
julia --project=. aastar_dqn.jl
```

<center><img src="https://github.com/bhatiaabhinav/Metareasoning.jl/blob/main/sp_example.gif" width="200"/></center>


----------------------------------------------

## Baselines

### RRT* 

In `rrt_baselines.jl`, set the desired growth factor, and run:

```bash
julia --project=. rrt_dqn.jl
```

### AWA*

Simply run:
```bash
julia --project=. aastar_baselines.jl
```
