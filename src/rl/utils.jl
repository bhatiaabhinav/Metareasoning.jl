using ReinforcementLearning
using Plots
using Random
using IntervalSets
using LinearAlgebra
using StatsBase
using Flux

export evaluate_policy, evaluate_policy_one_episode, record_clip

const AbstractSpace{N, T<:Union{Integer, AbstractFloat}} = Union{Space{Array{ClosedInterval{T}, N}}, ClosedInterval{T}, Base.OneTo{T}}

isdiscrete(s::Base.OneTo) = true
isdiscrete(s::AbstractSpace) = false

Base.size(s::AbstractInterval) = ()
Base.size(s::Base.OneTo{T}) where T = ()  # This is an override
Base.eltype(s::Space{Array{I, N}}) where {I, N} = eltype(I)
Base.eltype(s::Space{I}) where I = eltype(I)

function record_clip(env::AbstractEnv; policy::AbstractPolicy=RandomPolicy(), steps=100)
    reset!(env)
    plot(env)
    anim = @animate for i=1:steps
        a = policy(env)
        try
            @assert a ∈ action_space(env) "Invalid action. Action $a ∉ Action Space $(action_space(env))"
        catch
            @assert false "Invalid action. Action $a ∉ Action Space $(action_space(env))"
        end
        env(a)
        plot(env)
        if is_terminated(env)
            reset!(env)
            plot(env)
        end
    end
    return anim
end

@inline uniform_noise(noise_dim, mb_size) = 2f0 .* rand(Float32, noise_dim, mb_size) .- 1f0

@inline normal_noise(noise_dim, mb_size) = randn(Float32, noise_dim, mb_size)


function preprocess(obs::AbstractArray{T})::Array{Float32} where {T <: Real}
    if T == UInt8
        return convert(Array{Float32}, obs) ./ 255
    else
        return convert(Array{Float32}, obs)
    end
end

function l2norm(x)
    sqrt(sum(x.^2))
end

function clip_gradients!(grads, clip_by_norm::Real)
    if clip_by_norm < Inf
        for grad in grads
            if isnothing(grad)
                continue
            end
            norm = l2norm(grad)
            if norm > clip_by_norm
                grad .*= clip_by_norm / norm
            end
        end
    end
    return nothing
end


function linear_schedule(steps::Integer, over_steps::Integer, start_val::Real, final_val::Real)
    over_steps == 0 && return final_val
    return clamp(start_val + (final_val - start_val) * steps / over_steps, min(start_val, final_val), max(start_val, final_val))
end

softmax

"""yᵢ = exp(xᵢ/α) / (∑ⱼ exp(xⱼ/α)). When temperature α=0, yᵢ = 1 for argmax(x) otherwise 0."""
function softmax_with_temperature(x::AbstractVector{T}; α::T)::AbstractVector{T} where {T <: Real}
    return @views softmax_with_temperature(Flux.unsqueeze(x, 2); α=α)[:, 1]
end

"""Batch version of softmax_with_temperature"""
function softmax_with_temperature(x::AbstractMatrix{T}; α::T)::Matrix{T} where {T <: Real}
    if α > 0
        max_x::AbstractMatrix{T} = maximum(x, dims=1)
        _x::Matrix{T} = (x .- max_x) / T(α)
        exps::Matrix{T} = exp.(_x)
        return exps ./ sum(exps, dims=1)
    else
        p::Matrix{T} = zeros(T, size(x))
        p[argmax(x, dims=1)] .= T(1)
        return p
    end
end

function categorical_sample(rng, values, probability_weights)
    return sample(rng, values, ProbabilityWeights(probability_weights))
end

function boltzman_sample(rng, x::AbstractVector{T}; α::_T = 1)::Int where {T <: Real, _T <: Real}
    α = T(α)
    return α > 0 ? categorical_sample(rng, 1:length(x), softmax_with_temperature(x, α=α)) : argmax(x)
end

function boltzman_sample(x::AbstractVector{T}; α::_T = 1)::Int where {T <: Real, _T <: Real}
    return boltzman_sample(Random.GLOBAL_RNG, x; α=α)
end



@inline logit(x) = log(x) - log(1-x)


function evaluate_policy_one_episode(p::AbstractPolicy, env::AbstractEnv; seed::Union{Nothing, Integer}=nothing)::Float64
    !isnothing(seed) && Random.seed!(env, seed)
    reset!(env)
    R::Float64 = 0
    while !is_terminated(env)
        env |> p |> env
        R += reward(env)
    end
    return R
end


function evaluate_policy(p::AbstractPolicy, env::AbstractEnv; episodes::Integer=100, seed::Union{Nothing, Integer}=nothing, reseed_episodes::Bool=false)::Float64
    !isnothing(seed) && !reseed_episodes && Random.seed!(env, seed)
    R̄::Float64 = 0
    for ep_no in 1:episodes
        ep_seed::Union{Nothing, Int} = !isnothing(seed) && reseed_episodes ? seed + ep_no : nothing
        R = evaluate_policy_one_episode(p, env; seed=ep_seed)
        R̄ = ((ep_no - 1) * R̄ + R) / ep_no
    end
    return R̄
end



function with_no_exploration(f::Function, ::AbstractPolicy)
    f()
end