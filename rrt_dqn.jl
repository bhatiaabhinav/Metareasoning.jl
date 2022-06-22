using ReinforcementLearning
using Flux
using Zygote
using Plots
using Random
using Logging
using StatsBase
using BSON
using WandbMacros
using RRTStar
using Metareasoning
ENV["JULIA_NO_WANDB"] = false   # set to true if you want to disable wandb.

projectname = "RRTStarMetareasoning"
experimentname = "DQN"
logdir = "logs/$projectname/$experimentname"

env = RRTStarControlEnv(max_samples=1000, monitoring_interval=1000÷20,  # Each episode = 20 steps.
    α = 1, β = 0, allow_interrupt_action=false,                         # Contract setting
    initial_growth_factor=1.5, growth_factor_range=(0.5, 2.5),
    focus_level = 0.4, initial_focus_index = (2, 2)                     # 40% sampling bias towards the focus region
)

train = false # Set to false to only evaluate a previously trained model
if train
    rm(logdir, force=true, recursive=true)
    mkpath("$logdir/plots")
    mkpath("$logdir/clips")
    mkpath("$logdir/models")

    h = RLHyperparameters()
    h.max_episodes = 30000
    h.min_explore_steps = 30000
    h.ϵ_anneal_period = 30000
    h.ϵ = 0.1
    h.hidden_dims = (64, 32)
    h.minibatch_size = 256
    h.η = 0.0001
    dqn = DQN(env, h);

    @wandbinit project=projectname name=experimentname reinit=true
    @wandbconfig to_dict(h)...
    @wandbsave "$logdir/*"
    @wandbsave "$logdir/plots/*"
    @wandbsave "$logdir/clips/*"

    hook = DoEveryNEpisode() do n, p, env
        @wandblog step=p.housekeeping.steps env.info... to_dict(p.housekeeping, (:steps, :episodes, :episode_steps, :episode_return, :sgd_updates, :value_loss, :value, :ϵ))...
        if n % 500 == 0
            with_no_exploration(dqn) do
                eval_mean_utility = evaluate_policy(dqn, env, episodes=100)
                println("episodes: ", n, "\teval mean utility: ", eval_mean_utility)
                @wandblog step=p.housekeeping.steps eval_mean_utility
                gif(record_clip(env, policy=dqn, steps=100), "$logdir/clips/$n.gif", fps=10)
                n % 5000 == 0 && BSON.@save "$logdir/models/$n.bson" dqn
            end
        end
        return nothing
    end

    Random.seed!(env, 0)
    run(dqn, env, StopAfterEpisode(h.max_episodes), hook);

    savefig(plot(dqn.housekeeping.returns_history), "$logdir/plots/training_curve.png")

    println("Saving model $logdir/models/final_model.bson")
    BSON.@save "$logdir/models/final_model.bson" dqn
end



# --------------------- Evaluation ------------------------------------------

println("Loading model $logdir/models/final_model.bson")
BSON.@load "$logdir/models/final_model.bson" dqn

with_no_exploration(dqn) do
    Random.seed!(env, 42)  # Seed for testing

    println("Evaluating")
    Rs = [evaluate_policy_one_episode(dqn, env) for i in 1:1000]
    mkpath(logdir)
    open(f -> join(f, Rs, "\n"), "$logdir/final_scores.csv", "w")
    println("Mean utlity: ", mean(Rs))

    println("Making a clip")
    clip = record_clip(env, policy=dqn, steps=20 * 10)
    mkpath("$logdir/clips")
    mp4(clip, "$logdir/clips/final_video.mp4", fps=10)
    gif(clip, "$logdir/clips/final_video.gif", fps=10)
end

@wandbfinish
