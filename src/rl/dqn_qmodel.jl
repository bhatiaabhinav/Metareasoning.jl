using Flux
using Zygote

function make_dnn(size_in::Int, size_hidden::NTuple{N, Int}, size_out::Union{Int, Nothing}; σ::Function=relu, σ_out=identity, layernorm::Bool=false)::Chain where N
    layers = []
    x = size_in
    for h in size_hidden
        push!(layers, Dense(x, h, σ));
        if layernorm
            push!(layers, LayerNorm(h))
        end
        x = h
    end
    if size_out !== nothing
        push!(layers, Dense(x, size_out, σ_out))
    end
    net = Chain(layers...)
    return net
end

struct QModel{SN, NA, SNp1}
    net::Chain
    """Hiddens layers as specified"""
    function QModel(state_shape::Tuple, num_actions::Integer; hidden_dims::Tuple{Vararg{Int}}=(64, 32), layernorm=false, σ=relu)
        net = nothing
        SN = length(state_shape)
        NA = num_actions
        net = make_dnn(state_shape[1], hidden_dims, NA; σ=σ, layernorm=layernorm)
        return new{SN, NA, SN+1}(net)
    end
end

@Flux.functor QModel



@inline function (qmodel::QModel{SN, NA, SNp1})(states::AbstractArray{Float32, SNp1}) where {SN, NA, SNp1}
    return qmodel.net(states)
end

# Unbatched input
@inline function (qmodel::QModel{SN, NA, SNp1})(states::AbstractArray{Float32, SN}) where {SN, NA, SNp1}
    states |> Flux.unsqueeze(SNp1) |> qmodel |>  q->@view q[:, 1]
end


@inline function (qmodel::QModel)(states::AbstractArray{S}) where S<:AbstractFloat
    Zygote.@ignore @warn "Input eltype to Q-network will be converted to Float32." typeof(states) maxlog=1
    states = Zygote.@ignore Float32.(states)
    return qmodel(states)
end

@inline function (qmodel::QModel)(states::AbstractArray{UInt8})
    Zygote.@ignore @info "Inputs states with eltype UInt8 will be converted to Float32 and divided by 255." typeof(states) maxlog=1
    states = Zygote.@ignore @. Float32(states) / 255f0
    return qmodel(states)
end


@inline function dqn_loss_fn(qmodel::QModel, mb_s::AbstractArray, mb_q::AbstractArray{Float32, 2})::Float32
    q = qmodel(mb_s)
    return Flux.Losses.mse(q, mb_q)
end