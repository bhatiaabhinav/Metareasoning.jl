using ReinforcementLearning
using RRTStar
using Random
using Plots
using CSV
using FileIO
using Distributions
using StatsBase


include("focus_region_features.jl")

# The actions in the action space:
const RRTENV_ACTION_NOOP = 1
const RRTENV_ACTION_GROWTHFACTOR_INC = 2
const RRTENV_ACTION_GROWTHFACTOR_DEC = 3
const RRTENV_ACTION_FOCUS_UP = 4
const RRTENV_ACTION_FOCUS_DOWN = 5
const RRTENV_ACTION_FOCUS_LEFT = 6
const RRTENV_ACTION_FOCUS_RIGHT = 7
const RRTENV_ACTION_INTERRUPT = 8

const RRTENV_ACTION_GROWTHFACTOR_INC_AMT = 0.5


"""An RL environment that allows monitoring and adjusting the RRT* algorithm"""
mutable struct RRTStarControlEnv <: AbstractEnv
    # variables that specify the RL env:
    rrtstar::RRTStarPlanner
    obstacles_density_range::Tuple{Float64, Float64}
    obstacles_sizeratio_range::Tuple{Float64, Float64}
    initial_growth_factor::Float64
    growth_factor_range::Tuple{Float64, Float64}
    focus_level::Float64
    initial_focus_index::Tuple{Int, Int}
    focus_index_range::Tuple{Int, Int}
    monitoring_interval::Int  # i.e. in terms of num samples/iterations
    allow_interrupt_action::Bool
    α::Float64
    β::Float64
    map_rng::MersenneTwister  # for map generator

    # of current instance:
    obstacles_density::Float64
    obstacles_sizeratio::Float64
    optimal_cost_estimate::Float64
    current_utility::Float64
    reward::Float64         # latest reward
    state::Vector{Float32}  # latest state
    done::Bool              # whether episode terminated
    info::Dict
    action::Int             # latest action

    # To track history of adjustments for current instance:
    history_qualities::Vector{Float64}
    history_times::Vector{Float64}
    history_growth_factors::Vector{Float64}
    history_focus_indices_x::Vector{Int}
    history_focus_indices_y::Vector{Int}

    function RRTStarControlEnv(;x_range::Tuple{Real, Real} = (-5.0, 5.0), y_range::Tuple{Real, Real} = (-5.0, 5.0), start::Tuple{Real, Real} = (-4.8, 4.8), goal::Tuple{Real, Real} = (4.8, -4.8), max_samples::Integer = 1000, obstacles_density_range::Tuple{Real, Real} = (0.3, 0.4), obstacles_sizeratio_range::Tuple{Real, Real} = (0.03, 0.035), initial_growth_factor::Real=1.5, growth_factor_range::Tuple{Real, Real}=(0.5, 2.5), focus_level::Real=0.4, initial_focus_index::Tuple{Int, Int}=(2,2), focus_index_range::Tuple{Int, Int}=(1,4), monitoring_interval::Integer=50, allow_interrupt_action::Bool=false, α::Real=1, β::Real=0)
        rrtstar = RRTStarPlanner(x_range, y_range, start, goal, Set{RRTStar.AbstractObstacle}(), max_samples, growth_factor=initial_growth_factor, focus_level=focus_level, focus_index=initial_focus_index, goal_bias=0.01)
        env =  new(rrtstar, obstacles_density_range, obstacles_sizeratio_range, initial_growth_factor, growth_factor_range, focus_level, initial_focus_index, focus_index_range, monitoring_interval, allow_interrupt_action, α, β, MersenneTwister())
        env.obstacles_density = 0
        env.obstacles_sizeratio = 0
        env.optimal_cost_estimate = 0
        env.current_utility = 0
        env.history_qualities = Float64[]
        env.history_times = Float64[]
        env.history_growth_factors = Float64[]
        env.history_focus_indices_x = Int[]
        env.history_focus_indices_y = Int[]
        visualize_init(rrtstar)

        env.reward = 0
        env.state = Float32[]
        env.done = false
        env.info = Dict()
        env.action = 1

        return env
    end
end


"""C*/C, where C* is an underestimate of optimal cost"""
quality(env::RRTStarControlEnv) = env.optimal_cost_estimate / env.rrtstar.costs[env.rrtstar.goal]


Base.time(env::RRTStarControlEnv) = env.rrtstar.num_points_sampled / env.rrtstar.max_samples


"""The state space:"""
obs(env::RRTStarControlEnv) = convert(Vector{Float32}, vcat(
    quality(env),
    time(env),
    env.rrtstar.growth_factor,
    (env.rrtstar.num_total_points + env.rrtstar.new_batch_size) / env.rrtstar.num_points_sampled,  # hit ratio
    env.rrtstar.dist_to_goal_lb / env.optimal_cost_estimate,
    log2(1 + length(extract_best_path(env.rrtstar))),
    env.obstacles_density,
    env.obstacles_sizeratio,
    env.rrtstar.focus_level,
    env.rrtstar.focus_index[1],
    env.rrtstar.focus_index[2],
    vec(env.rrtstar.focus_area_scores),
    ))

"""Action space"""
function RLBase.action_space(env::RRTStarControlEnv)
    if !env.allow_interrupt_action
        return Base.OneTo(7)  # Adjust growth_factor, focus_area
    else
        return Base.OneTo(8)
    end
end

function RLBase.state_space(env::RRTStarControlEnv)
    n = length(obs(env))
    return Space(fill(ClosedInterval{Float32}(0, 1), n))
end

function Random.seed!(env::RRTStarControlEnv, seed)
    Random.seed!(env.map_rng, seed)
    Random.seed!(env.rrtstar, seed)
end

RLBase.reward(env::RRTStarControlEnv) = env.reward

RLBase.state(env::RRTStarControlEnv) = env.state

RLBase.is_terminated(env::RRTStarControlEnv) = env.done


"""Reset the data structures of this RL environment and launch RRT* on a new map instance."""
function RLBase.reset!(env::RRTStarControlEnv)
    env.obstacles_density = rand(env.map_rng, Uniform(env.obstacles_density_range...))
    env.obstacles_sizeratio = rand(env.map_rng, Uniform(env.obstacles_sizeratio_range...))
    env.rrtstar.obstacles = rand_rect_obstacles(env.map_rng, env.rrtstar.x_range, env.rrtstar.y_range, env.rrtstar.start, env.rrtstar.goal, rect_density=env.obstacles_density, rect_sizeratio=env.obstacles_sizeratio)
    env.rrtstar.growth_factor = env.initial_growth_factor
    env.rrtstar.focus_level = env.focus_level
    env.rrtstar.focus_index = env.initial_focus_index
    RRTStar.reset!(env.rrtstar)

    env.optimal_cost_estimate = l2dist(env.rrtstar.start, env.rrtstar.goal)
    q, t = quality(env), time(env)
    env.current_utility = time_dependent_utility(q, t, env.α, env.β)

    empty!(env.history_qualities)
    empty!(env.history_times)
    empty!(env.history_growth_factors)
    empty!(env.history_focus_indices_x)
    empty!(env.history_focus_indices_y)

    env.reward = 0
    env.state = obs(env)
    env.done = false
    env.info = Dict()

    return nothing
end


"""This is the step of this RL environment: The core logic for adjusting the hyperparameters of RRT* algorithm based on the action taken by the RL agent."""
function (env::RRTStarControlEnv)(action::Integer)
    @assert action ∈ action_space(env)
    env.action = action
    done::Bool = false
    interrupted::Bool = false
    num_points_accepted::Int = env.rrtstar.num_total_points + env.rrtstar.new_batch_size

    @debug "Stepping" action
    if action == RRTENV_ACTION_INTERRUPT
        @debug "Interrupt action" action
        done = true
        interrupted = true
    else
        if action == RRTENV_ACTION_NOOP
            @debug "NOOP action" action
        elseif action == RRTENV_ACTION_GROWTHFACTOR_INC
            env.rrtstar.growth_factor = clamp(env.rrtstar.growth_factor + RRTENV_ACTION_GROWTHFACTOR_INC_AMT, env.growth_factor_range[1], env.growth_factor_range[2])
        elseif action == RRTENV_ACTION_GROWTHFACTOR_DEC
            env.rrtstar.growth_factor = clamp(env.rrtstar.growth_factor - RRTENV_ACTION_GROWTHFACTOR_INC_AMT, env.growth_factor_range[1], env.growth_factor_range[2])
        elseif action == RRTENV_ACTION_FOCUS_UP
            env.rrtstar.focus_index = clamp.(env.rrtstar.focus_index .+ (0, 1), env.focus_index_range[1], env.focus_index_range[2])
        elseif action == RRTENV_ACTION_FOCUS_DOWN
            env.rrtstar.focus_index = clamp.(env.rrtstar.focus_index .+ (0, -1), env.focus_index_range[1], env.focus_index_range[2])
        elseif action == RRTENV_ACTION_FOCUS_LEFT
            env.rrtstar.focus_index = clamp.(env.rrtstar.focus_index .+ (-1, 0), env.focus_index_range[1], env.focus_index_range[2])
        elseif action == RRTENV_ACTION_FOCUS_RIGHT
            env.rrtstar.focus_index = clamp.(env.rrtstar.focus_index .+ (1, 0), env.focus_index_range[1], env.focus_index_range[2])
        else
            @error "Unrecognized action" action
            error("Unrecognized action $(action)")
        end

        @debug "Continuing execution" env.rrtstar.num_points_sampled env.rrtstar.growth_factor env.rrtstar.focus_index env.rrtstar.focus_level
        search!(env.rrtstar, sample_limit=env.monitoring_interval)
        if env.rrtstar.num_points_sampled >= env.rrtstar.max_samples
            @debug "Natural done"
            done = true
        end
        num_points_accepted = env.rrtstar.num_total_points + env.rrtstar.new_batch_size
    end
        
    q, t, g, f = quality(env), time(env), env.rrtstar.growth_factor, env.rrtstar.focus_level
    push!(env.history_qualities, q)
    push!(env.history_times, t)
    push!(env.history_growth_factors, g)
    push!(env.history_focus_indices_x, env.rrtstar.focus_index[1])
    push!(env.history_focus_indices_y, env.rrtstar.focus_index[2])
    compute_focus_area_scores(env.rrtstar)
    obs_next = obs(env)
    prev_utility = env.current_utility
    env.current_utility = time_dependent_utility(q, t, env.α, env.β)
    reward = env.current_utility - prev_utility
    
    if done
        env.info = Dict(
            :quality => q,
            :cost => env.rrtstar.costs[env.rrtstar.goal],
            :time => t,
            :utility => env.current_utility,
            :interrupted => Int(interrupted),
            :growth_factor_av => mean(env.history_growth_factors),
            :focus_index_x_av => mean(env.history_focus_indices_x),
            :focus_index_y_av => mean(env.history_focus_indices_y),
            :num_points_sampled => env.rrtstar.num_points_sampled,
            :num_points_accepted => num_points_accepted,
            :num_points_in_solution => length(extract_best_path(env.rrtstar)),
            :hit_ratio => num_points_accepted / env.rrtstar.num_points_sampled
        )
    end

    @debug "Step finish" reward done info...
    env.state .= obs_next
    env.reward = reward
    env.done = done
    return nothing
end


"""For the current episode"""
function plot_quality_vs_time(env::RRTStarControlEnv)
    pl = plot(env.history_times, env.history_qualities, title="Quality, Growth-Factor, and Focus Index vs Time", label="q", legend=:topleft)
    plot!(env.history_times, env.history_growth_factors, label="g")
    plot!(env.history_times, env.history_focus_indices_x, label="x")
    plot!(env.history_times, env.history_focus_indices_y, label="y")
    xlabel!("Time")
    return pl
end


function Plots.plot(env::RRTStarControlEnv)
    plot(visualize_frame(env.rrtstar), xtick=nothing, ytick=nothing, border=:none)
end
