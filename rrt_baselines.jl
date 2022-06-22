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

projectname = "RRTStarMetareasoning"

gf = 1.0  # Small
# gf = 2.0  # Large
experimentname = "StaticGrowthFactor=$gf"
logdir = "logs/$projectname/$experimentname"

env = RRTStarControlEnv(max_samples=1000, monitoring_interval=1000÷20,  # Each episode = 20 steps.
    α = 1, β = 0, allow_interrupt_action=false,
    initial_growth_factor=gf, growth_factor_range=(gf, gf),             # no wiggle room to adjust growth factor
    focus_level = 0.0                                                   # no bias
)


# --------------------- Evaluation ------------------------------------------

Random.seed!(env, 42)  # Seed for testing

println("Evaluating")
Rs = [evaluate_policy_one_episode(RandomPolicy(), env) for i in 1:1000]
mkpath(logdir)
open(f -> join(f, Rs, "\n"), "$logdir/final_scores.csv", "w")
println("Mean utlity: ", mean(Rs))

println("Making a clip")
clip = record_clip(env, policy=RandomPolicy(), steps=20 * 10)
mkpath("$logdir/clips")
mp4(clip, "$logdir/clips/final_video.mp4", fps=10)
gif(clip, "$logdir/clips/final_video.gif", fps=10)

