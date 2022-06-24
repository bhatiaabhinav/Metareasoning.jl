using ReinforcementLearning
using Random
using Zygote: Grads, Zygote
using Flux: params, cpu, gpu, Flux
using CUDA
using Flux.Optimise
using Plots
using StatsBase
import BSON

export DQN, with_no_exploration

"""Double Deep Q Network (Mnih et al. 2015, Van Hasselt et al. 2016)"""
mutable struct DQN{ST, SN, NA, R, SNp1} <: AbstractPolicy  # State eltype, state_ndims, num actions
    hyperparams::RLHyperparameters

    housekeeping::Housekeeping{Array{ST, SN}, Int, R}
    exp_buff::ExperienceBuffer{ST, SN, Int, 0, R, SNp1, 1}
    qmodel::QModel{SN, NA, SNp1}
    ϕ::Params                                           # Parameters of QModel
    target_qmodel::QModel{SN, NA, SNp1}
    optimizer::ADAM

    # For sampling from experience buffer, preallocated memory for minibatches:
    buff_s::Array{Float32, SNp1}
    buff_a::Vector{Int}
    buff_r::Vector{Float32}
    buff_s′::Array{Float32, SNp1}
    buff_s′∞::Array{Bool}                               # Whether s′ is absorbing

    function DQN(env::AbstractEnv, hyperparams::RLHyperparameters)
        h = hyperparams
        
        ss = state_space(env)
        as = action_space(env)
        state_shape = size(ss)
        ST = eltype(state_space(env)[1])
        R = typeof(reward(env))  
        SN = length(state_shape)
        NA = length(as)

        r = Housekeeping{Array{ST, SN}, Int, R}(h)
        r.s = zeros(ST, state_shape...) # preallocate memory
        r.s′ = zeros(ST, state_shape...)

        exp_buff = ExperienceBuffer(h.experience_buffer_size, ST, state_shape, Int, (), R)
        device = h.device
        qmodel = QModel(state_shape, NA; hidden_dims=h.hidden_dims, σ=h.σ, layernorm=h.layernorm)
        target_qmodel = QModel(state_shape, NA; hidden_dims=h.hidden_dims, σ=h.σ, layernorm=h.layernorm)
        ϕ = params(qmodel)
        Flux.loadparams!(target_qmodel, ϕ)
        optimizer = ADAM(h.η)
        buff_s = zeros(ST, state_shape..., h.minibatch_size)
        buff_a = zeros(Int, h.minibatch_size)
        buff_r = zeros(Float32, h.minibatch_size)
        buff_s′ = zeros(ST, state_shape..., h.minibatch_size)
        buff_s′∞ = zeros(Bool, h.minibatch_size)
        @info "DQN" ss as device qmodel optimizer typeof(exp_buff)

        return new{ST, SN, NA, R, SN+1}(h, r, exp_buff, qmodel, ϕ, target_qmodel, optimizer, buff_s, buff_a, buff_r, buff_s′, buff_s′∞)
    end
end


function (d::DQN)(::PreExperimentStage, env::AbstractEnv)
    # Don't re-initialize so that we can re-`run` to resume training.
    return nothing
end

function (d::DQN)(::PreEpisodeStage, env::AbstractEnv)
    @debug "In PreEpisodeStage"
    d.housekeeping.episode_return = 0
    d.housekeeping.episode_steps = 0
    if !is_trajectory_terminated(d.exp_buff)
        terminate_trajectory!(d.exp_buff)
    end
    return nothing
end

function (d::DQN{ST, SN, NA})(env::AbstractEnv) where {ST, SN, NA}
    @debug "returning dqn act"
    if (d.housekeeping.steps < d.hyperparams.min_explore_steps || (d.housekeeping.ϵ > 0 && rand(d.housekeeping.exploration_rng) < d.housekeeping.ϵ))
        return rand(d.housekeeping.exploration_rng, 1:NA)
    else
        qvalues = env |> state |> d.qmodel
        return boltzman_sample(qvalues, α=0.001)  # A little "soft" argmax
    end
end

function with_no_exploration(f::Function, dqn::DQN)
    cur_ϵ  = dqn.housekeeping.ϵ
    return_val = f()
    dqn.housekeeping.ϵ = cur_ϵ
    return return_val
end


function (d::DQN)(::PreActStage, env::AbstractEnv, a)
    d.housekeeping.s .= state(env)
    d.housekeeping.a = a
    @debug "in pre act" a
    return nothing
end



function (d::DQN)(::PostActStage, env::AbstractEnv)
    @debug "in post act"
    d.housekeeping.r = reward(env)
    d.housekeeping.s′ .= state(env)
    d.housekeeping.is_terminated = is_terminated(env)
    d.housekeeping.steps += 1
    d.housekeeping.episode_steps += 1
    d.housekeeping.episode_return += d.housekeeping.r
    d.housekeeping.is_s′_absorbing = d.housekeeping.is_terminated && d.housekeeping.episode_steps < d.hyperparams.env_fixedhorizon_steps

    push!(d.exp_buff, d.housekeeping.s, d.housekeeping.a, d.housekeeping.r)
    if d.housekeeping.is_terminated
        terminate_trajectory!(d.exp_buff, d.housekeeping.s′, d.housekeeping.is_s′_absorbing)
    end

    if d.housekeeping.steps >= d.hyperparams.min_explore_steps && d.housekeeping.steps % d.hyperparams.train_interval == 0
        dqn_train_from_buffer!(d, d.exp_buff, 1)
    end

    d.housekeeping.ϵ = linear_schedule(d.housekeeping.steps - d.hyperparams.min_explore_steps, d.hyperparams.ϵ_anneal_period, 1, d.hyperparams.ϵ)

    return nothing
end

function (d::DQN)(::PostEpisodeStage, env::AbstractEnv)
    @debug "in post episode"
    d.housekeeping.episodes += 1
    push!(d.housekeeping.returns_history, d.housekeeping.episode_return)
    return nothing
end

function get_action_probabilities(q::Matrix{Float32}, num_actions::Int, ϵ::Float32)::Matrix{Float32}
    πa::Matrix{Float32} = fill(ϵ / num_actions, size(q))
    πa[argmax(q, dims=1)] .+= 1 - ϵ
    return πa
end


function dqn_train_from_buffer!(d::DQN{ST, SN, NA, R, SNp1}, buffer::ExperienceBuffer{ST, SN, Int, 0, R, SNp1, 1}, sgd_steps::Integer) where {ST, SN, NA, R, SNp1}
    ℓϕ::Float32, value::Float32, entropy::Float32 = 0.0, 0.0, 0.0
    ϵ::Float32 = d.housekeeping.ϵ  # epsilon of the behavior policy
    ϵ′::Float32 = d.hyperparams.sarsa ? ϵ : 0f0  # eplison of the target policy.
    ρ::Float32 = d.hyperparams.ρ
    mb_size::Int = d.hyperparams.minibatch_size

    s::Array{Float32, SNp1}, a::Vector{Int}, r::Vector{Float32}, s′::Array{Float32, SNp1}, s′∞::Vector{Bool} = d.buff_s, d.buff_a, d.buff_r, d.buff_s′, d.buff_s′∞

    for step_no in 1:sgd_steps
        sample_experiences!(d.housekeeping.learning_rng, buffer, mb_size, s, a, r, s′, nothing, s′∞)  # in place sampling

        q::Matrix{Float32} = d.qmodel(s)
        # ------ Record average value of minibatch states under the current policy ----------
        πa::Matrix{Float32} = get_action_probabilities(q, NA, ϵ′)  # probability of actions on s under the current policy
        value = mean(sum(πa .* q, dims=1))  # v(s) = ∑ q(s,a) π(s,a)
        entropy = mean(sum(-πa .* log.(πa .+ 1f-9), dims=1))
        # -----------------------------------------------------------------------------------

        πa′::Matrix{Float32} = get_action_probabilities(d.qmodel(s′), NA, ϵ′)  # probability of actions on s′ under the current policy
        v′::Vector{Float32} = sum(πa′ .* d.target_qmodel(s′), dims=1)[1, :]  # Value of s′ under the current policy
        y::Vector{Float32} = r + d.hyperparams.γ * ((1f0 .- s′∞) .* v′)  # These should be the q-values of (s,a)
        for i in 1:mb_size
            q[a[i], i] = y[i]  # Modify the current q values appropriately for each state in the minibatch
        end

        ℓϕ, ∇ϕ::Grads = withgradient(d.ϕ) do
            dqn_loss_fn(d.qmodel, s, q)
        end
        clip_gradients!(∇ϕ, d.hyperparams.grad_clipping)
        Optimise.update!(d.optimizer, d.ϕ, ∇ϕ)
        d.housekeeping.sgd_updates += 1
        d.housekeeping.sgd_updates % d.hyperparams.Δτ == 0 && Flux.loadparams!(d.target_qmodel, ρ .* params(d.target_qmodel) .+ (1 - ρ) .* d.ϕ)
    end
    d.housekeeping.value, d.housekeeping.value_loss, d.housekeeping.entropy = value, ℓϕ, entropy
    return nothing
end
