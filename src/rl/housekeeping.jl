export Housekeeping

"""A struct for housekeeping"""
mutable struct Housekeeping{S,A,R}
    steps::Int  # Since beganing
    episodes::Int
    episode_steps::Int
    episode_return::Float32
    sgd_updates::Int
    value_loss::Float32  # average loss functions
    value::Float32  # "value" of an average state given current policy
    entropy::Float32  # average entropy of current policy
    ϵ::Float32  # current value
    
    # Random Number Generators. Xoshiro is known to be a pretty fast one.
    exploration_rng::Xoshiro
    learning_rng::Xoshiro

    # latest experience:
    s::Union{Nothing, S}
    a::Union{Nothing, A}
    r::Union{Nothing, R}
    s′::Union{Nothing, S}
    is_terminated::Bool
    is_s′_absorbing::Bool

    returns_history::Vector{Float32}

    misc::Dict{Symbol, Any}  # Those not covered above

    function Housekeeping{S,A,R}(h::RLHyperparameters) where {S, A, R}
        return new{S,A,R}(0, 0, 0, 0, 0, 0, 0, 0, h.ϵ,
                    Xoshiro(h.exploration_seed), Xoshiro(h.learning_seed),
                    nothing, nothing, nothing, nothing, false, false,
                    Float32[],
                    Dict{Symbol, Any}())
    end
end

function to_dict(p::Housekeeping, keys::Tuple{Vararg{Symbol}})
    d = Dict{Symbol, Any}()
    for k in keys
        if haskey(p.misc, k)
            d[k] = p.misc[k]
        elseif hasfield(Housekeeping, k)
            d[k] = getfield(p, k)
        else
            throw(KeyError(k))
        end
    end
    return d
end