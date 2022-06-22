using ReinforcementLearning
using IntervalSets
using AnytimeWeightedAStar: AbstractWeightAdjustmentPolicy, AnytimeWeightedAStar, possible_weights, AWAStar, simulated_time, expansion_rate
using AnytimeWeightedAStar.SearchProblem
using AnytimeWeightedAStar.SearchProblem:AbstractSearchProblem
using AnytimeWeightedAStar.GraphSearch: init!, graph_search!, node_expansion_policy, before_node_expansion!, get_children_nodes, on_node_generation!, on_node_expansion_finish!, on_stop!, stop_condition
using Random
using Plots
using LaTeXStrings


# The actions in the action space:
const AWAENV_ACTION_NOOP = 1
const AWAENV_ACTION_WEIGHT_INC = 2
const AWAENV_ACTION_WEIGHT_DEC = 3
const AWAENV_ACTION_WEIGHT_JUMP_UP = 4
const AWAENV_ACTION_WEIGHT_JUMP_DOWN = 5
const AWAENV_ACTION_INTERRUPT = 6  # Make sure interrupt action is last

const AWAENV_ACTION_WEIGHT_INC_AMT = 1  # index increase
const AWAENV_ACTION_WEIGHT_JUMP_AMT = 3  # index increase


mutable struct CustomWeightAdjustmentPolicy <: AbstractWeightAdjustmentPolicy
    weight_set::Vector{Float64}
    current_index::Int
end


AnytimeWeightedAStar.possible_weights(cwap::CustomWeightAdjustmentPolicy) = cwap.weight_set
function (cwap::CustomWeightAdjustmentPolicy)(;kwargs...)
    return cwap.weight_set[cwap.current_index]
end


"""An RL environment that executes and monitors AWAStar. If there is a node_budget, time is measured as nodes_expanded/reference_nodeexpansion_rate."""
mutable struct AWAStarControlEnv{SearchProblem <: AbstractSearchProblem} <: AbstractEnv
    # Should be supplied to constructor:
    search_problem::SearchProblem
    weight_set::Vector{Float64}
    monitoring_interval::Int    # In terms of node_expansions
    allow_interrupt_action::Bool 
    α::Float64  # quality coefficient in time_dependent_utility
    β::Float64  # cost of time = e^(βt). -inf corresponds to no time cost.
    node_budget::Union{Int,Nothing}  # "Time" is measured as nodes_expanded/node_budget
    
    # contructed objects:
    wap::CustomWeightAdjustmentPolicy
    awastar::AWAStar

    # episode-specific variables:
    sp_optimal_cost::Float64  # optimal cost of current search problem
    current_utility::Float64
    last_step_at::Float64  # time
    reward::Float64   # current reward
    state::Vector{Float32}  # current state
    done::Bool  # whether terminated
    info::Dict
    action::Int

    # records
    qualities_history::Vector{Float64}
    times_history::Vector{Float64}
    weights_history::Vector{Float64}
    quality_upperbounds_history::Vector{Float64}

    function AWAStarControlEnv(search_problem::AbstractSearchProblem{S, A}; weight_set::Vector{T}, node_budget::Int, monitoring_interval::Int, allow_interrupt_action::Bool, α::Real, β::Real) where {S, A,T<:Real}
        env::AWAStarControlEnv  = new{typeof(search_problem)}(search_problem, weight_set, monitoring_interval, allow_interrupt_action, α, β, node_budget)
        env.wap = CustomWeightAdjustmentPolicy(weight_set, 1)
        env.awastar = AWAStar{S, A}(1, Inf, env.node_budget, env.wap)
        env.qualities_history = Float64[]
        env.times_history = Float64[]
        env.weights_history = Float64[]
        env.quality_upperbounds_history = Float64[]

        env.reward = 0   # current reward
        env.state = Float32[]
        env.done = false
        env.info = Dict()
        env.action = 1

        return env
    end
end


"""C*/C, where C* is an underestimate of optimal cost"""
quality(env::AWAStarControlEnv) = AnytimeWeightedAStar.quality(env.awastar.solution_cost, env.sp_optimal_cost)


"""optimal cost divided by cost lower bound, which is the least f-value in the open list."""
function quality_upperbound(env::AWAStarControlEnv)
    if length(env.awastar.open_lists) > 0
        _, best_g, best_h  = peek(env.awastar.open_lists, 1.0)
    else
        best_g, best_h = 0, env.sp_optimal_cost
    end
    return env.sp_optimal_cost / (best_g + best_h)
end


"""wall time if node_budget is nothing, otherwise nodes_expanded/reference_nodeexpansion_rate"""
Base.time(env::AWAStarControlEnv) = env.awastar.nodes_expended / env.node_budget



"""The observation space constains relevant features for an RL agent to make decisions."""
function obs(env::AWAStarControlEnv)
    if length(env.awastar.open_lists) > 0
        _, best_g, best_h  = peek(env.awastar.open_lists, 1.0)
    else
        best_g, best_h = 0, env.sp_optimal_cost
    end

    if length(env.qualities_history) <= 20
        qualities_history_padded = vcat(reverse(env.qualities_history), zeros(20 - length(env.qualities_history)))
        weights_history_padded = vcat(reverse(env.weights_history), zeros(20 - length(env.weights_history)))
        quality_upperbounds_history_padded = vcat(reverse(env.quality_upperbounds_history), zeros(20 - length(env.qualities_history)))
    else
        qualities_history_padded = reverse(env.qualities_history)[1:20]
        weights_history_padded = reverse(env.weights_history)[1:20]
        quality_upperbounds_history_padded = reverse(env.quality_upperbounds_history)[1:20]
    end
    
    return convert(Array{Float32}, vcat(
        time(env),
        qualities_history_padded,
        weights_history_padded ./ maximum(possible_weights(env.wap)),
        quality_upperbounds_history_padded,
        collect(env.sp_optimal_cost ./ (env.awastar.open_lists.stats[1:4] .+ 1)),  # μ_g, μ_h, σ_g,, σ_h
        env.awastar.open_lists.stats[5],
        log10(env.awastar.open_lists.stats[6] + 1), # ρ_gh, log(n)
        env.sp_optimal_cost ./ ([best_g, best_h] .+ 1), # min_g, min_h, ̲g, ̲h
        SearchProblem.obs(env.search_problem)  # should include h₀
        ))
end

"""Action space"""
RLBase.action_space(env::AWAStarControlEnv) = Base.OneTo(env.allow_interrupt_action ? 6 : 5) # NOOP, INCR, DECR, INC_JUMP, DEC_JUMP, INTERRUPT

function RLBase.state_space(env::AWAStarControlEnv)
    l = length(SearchProblem.obs(env.search_problem))
    n = 69 + l
    return Space(fill(ClosedInterval{Float32}(0, 1), n))
end

function Random.seed!(env::AWAStarControlEnv, seed)
    Random.seed!(env.search_problem, seed)
end

RLBase.reward(env::AWAStarControlEnv) = env.reward

RLBase.state(env::AWAStarControlEnv) = env.state

RLBase.is_terminated(env::AWAStarControlEnv) = env.done

"""Reset the data structures of this RL environment and launch AWA* on a new search problem instance"""
function RLBase.reset!(env::AWAStarControlEnv)
    empty!(env.qualities_history)
    empty!(env.times_history)
    empty!(env.weights_history)
    empty!(env.quality_upperbounds_history)
    SearchProblem.reset!(env.search_problem)
    env.wap.current_index = 1
    env.sp_optimal_cost = heuristic(env.search_problem, start_state(env.search_problem))

    init!(env.awastar, env.search_problem)

    q, t = quality(env), time(env)
    env.current_utility = time_dependent_utility(q, t, env.α, env.β)
    env.reward = 0
    @debug q t env.α env.β env.current_utility
    env.last_step_at = t
    env.state = obs(env)
    env.done = false
    env.info = Dict()
    return nothing
end

"""steps env.awastar and returns whether the run finished"""
function awa_step!(env::AWAStarControlEnv)
    if !stop_condition(env.awastar, env.search_problem)
        node = node_expansion_policy(env.awastar, env.search_problem)
        before_node_expansion!(env.awastar, node, env.search_problem)
        for child_node in get_children_nodes(env.search_problem, node)
            on_node_generation!(env.awastar, child_node, env.search_problem)
        end
        on_node_expansion_finish!(env.awastar, node, env.search_problem)
        return false
    else
        on_stop!(env.awastar, env.search_problem)
        return true
    end
end

"""This is the step of this RL environment: The core logic for adjusting the weight of the awastar algorithm based on the action taken by the RL agent."""
function (env::AWAStarControlEnv)(action::Integer)
    @assert action ∈ action_space(env)
    env.action = action
    done = false
    interrupted = false
    prev_utility = env.current_utility

    @debug "Stepping" action
    if action == AWAENV_ACTION_INTERRUPT
        @debug "Interrupt action" action
        AnytimeWeightedAStar.interrupt!(env.awastar)
        done = awa_step!(env)
        @assert done "somehow awastar didn't stop despite interruption"
        interrupted = true
    else
        if action == AWAENV_ACTION_WEIGHT_INC
            env.wap.current_index = clamp(env.wap.current_index + AWAENV_ACTION_WEIGHT_INC_AMT, 1, length(env.wap.weight_set))
        elseif action == AWAENV_ACTION_WEIGHT_DEC
            env.wap.current_index = clamp(env.wap.current_index - AWAENV_ACTION_WEIGHT_INC_AMT, 1, length(env.wap.weight_set))
        elseif action == AWAENV_ACTION_WEIGHT_JUMP_UP
            env.wap.current_index = clamp(env.wap.current_index + AWAENV_ACTION_WEIGHT_JUMP_AMT, 1, length(env.wap.weight_set))
        elseif action == AWAENV_ACTION_WEIGHT_JUMP_DOWN
            env.wap.current_index = clamp(env.wap.current_index - AWAENV_ACTION_WEIGHT_JUMP_AMT, 1, length(env.wap.weight_set))
        elseif action == AWAENV_ACTION_NOOP
            @debug "NOOP action" env.wap()
        else
            @error "Unrecognized action" action
            error("Unrecognized action $(action)")
            end

        @debug "Continuing execution"
        while time(env) - env.last_step_at < (env.monitoring_interval / env.node_budget)
            done = awa_step!(env)
            if done
                @debug "natural done"
                break
            end
        end

        @debug "AWAStar executed for another delta time" (time(env) - env.last_step_at) (env.monitoring_interval / env.node_budget) done interrupted
    end

    q = quality(env)
    t = time(env)
    w = env.awastar.weight
    q_ub = quality_upperbound(env)
    push!(env.qualities_history, q)
    push!(env.times_history, t)
    push!(env.weights_history, w)
    push!(env.quality_upperbounds_history, q_ub)
    w_av = sum(env.weights_history) / length(env.weights_history)
    env.current_utility = time_dependent_utility(q, t, env.α, env.β)
    env.last_step_at = t
    reward = env.current_utility - prev_utility
    
    if done
        env.info = Dict(
            :quality => q,
            :cost => env.awastar.solution_cost
            :time => t,
            :utility => env.current_utility,
            :interrupted => Int(interrupted),
            :w_av => w_av,
            :nodes_expanded => env.awastar.nodes_expended,
            :num_solutions=>env.awastar.num_solutions)
        merge!(info, SearchProblem.info(env.search_problem))
    end

    @debug "Step finish" reward done info...
    env.state = obs(env)
    env.reward = reward
    env.done = done
    return nothing
end


function plot_quality_vs_time(env::AWAStarControlEnv)
    pl = plot(env.times_history, 10 .* env.qualities_history, label=L"10q", legend=:topleft)
    plot!(pl, env.times_history, 10 .* env.quality_upperbounds_history, label=L"10\hat{q}")
    plot!(pl, env.times_history, env.weights_history, label=L"w")
    w_av = sum(env.weights_history) / length(env.weights_history)
    plot!(pl, env.times_history, w_av .* ones(length(env.weights_history)), label=L"\mu_w")
    xlabel!(pl, "t")
    return pl
end


function Plots.plot(env::AWAStarControlEnv)
    plot_quality_vs_time(env)
end
