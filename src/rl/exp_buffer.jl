using Random

"""Assumes that the data comes in sequence"""
mutable struct ExperienceBuffer{ST, SN, AT, AN, R, SNp1, ANp1} # SN-Dim states of eltype ST. AN-Dim actions of eltype AT. Rewards of type R. SNp1=SN+1, ANp1=AN+1
    capacity::Int               # Should not be modified after creating an experience buffer!
    states::Array{ST, SNp1}     # Compatible with discrete Integer states as well for tabular RL. e.g., state_size=() and state_eltype = Int. ST can even be a symbol.
    actions::Array{AT, ANp1}    # Compatible with both discrete and continous action spaces.
    rewards::Vector{R}          # Allows storing costs as well. e.g., R=Tuple{Float32, Float32}.
    isterminals::Vector{Bool}   # whether is_terminated(env) is true. Note: true is not necessarily due to transition to an absorbing state (e.g., when an episode is cutoff to a timelimit).
    isabsorbing::Vector{Bool}   # Absorbing states (s_∞) are terminal states that have zero value function.
    _idx::Int                   # num states added so far.

    function ExperienceBuffer(capacity::Integer, state_eltype::Type{ST}, state_size::NTuple{SN, Int}, action_eltype::Type{AT}, action_size::NTuple{AN, Int}, reward_type::Type{R}) where {ST, SN, AT, AN, R}
        states = Array{ST}(undef, state_size..., capacity)
        actions = Array{AT}(undef, action_size..., capacity)
        rewards = Array{R}(undef, capacity)
        isterminals = zeros(Bool, capacity)
        isabsorbing = zeros(Bool, capacity)
        _idx = 0
        return new{ST, SN, AT, AN, R, SN+1, AN+1}(capacity, states, actions, rewards, isterminals, isabsorbing, _idx)
    end
end

function Base.empty!(buff::ExperienceBuffer)
    buff._idx = 0
    return nothing
end

Base.length(buff::ExperienceBuffer) = min(buff._idx, buff.capacity)

circularindex(entry_idx::Int, capacity::Int)::Int = (entry_idx - 1) % capacity + 1

function Base.push!(buff::ExperienceBuffer{ST, SN, AT, AN, R}, state::Union{_S, AbstractArray{_S, SN}}, action::Union{_A, AbstractArray{_A, AN}}, reward::_R) where {ST, SN, AT, AN, R, _S, _A, _R}
    buff._idx = buff._idx + 1
    i::Int = circularindex(buff._idx, buff.capacity)

    if state isa AbstractArray
        selectdim(buff.states, SN + 1, i) .= state
    else
        buff.states[i] = state
    end
    if action isa AbstractArray
        selectdim(buff.actions, AN + 1, i) .= action
    else
        buff.actions[i] = action
    end
    buff.rewards[i] = reward
    buff.isterminals[i] = false
    buff.isabsorbing[i] = false
    return nothing
end

function terminate_trajectory!(buff::ExperienceBuffer{ST, SN, AT, AN, R}, nextstate::Union{_S, AbstractArray{_S, SN}}, isabsorbing::Bool) where {ST, SN, AT, AN, R, _S}
    buff._idx = buff._idx + 1
    i::Int = circularindex(buff._idx, buff.capacity)

    if nextstate isa AbstractArray
        selectdim(buff.states, SN + 1, i) .= nextstate
    else
        buff.states[i] = nextstate
    end
    buff.isterminals[i] = true
    buff.isabsorbing[i] = isabsorbing

    return nothing
end


function terminate_trajectory!(buff::ExperienceBuffer{ST, SN, AT, AN, R}) where {ST, SN, AT, AN, R, _S}
    if length(buff) > 0
        i::Int = circularindex(buff._idx, buff.capacity)
        buff.isterminals[i] = true
    end
    return nothing
end

@inline function is_trajectory_terminated(buff::ExperienceBuffer{ST, SN, AT, AN, R}) where {ST, SN, AT, AN, R, _S}
    return length(buff) == 0 || buff.isterminals[circularindex(buff._idx, buff.capacity)]
end


function sample_states(rng::AbstractRNG, buff::ExperienceBuffer{ST, SN, AT, AN, R, SNp1, ANp1}, num_samples::Int) where {ST, SN, AT, AN, R, SNp1, ANp1}
    dest_states = Array{ST}(undef, size(buff.states)[1:end-1]..., num_samples)
    sample_experiences!(rng, buff, num_samples, dest_states)
    return dest_states
end

function sample_states(buff::ExperienceBuffer, args...)
    return sample_states(Random.GLOBAL_RNG, buff, args...)
end

function sample_state(rng::AbstractRNG, buff::ExperienceBuffer{ST, SN, AT, AN, R, SNp1, ANp1}) where {ST, SN, AT, AN, R, SNp1, ANp1}
    return selectdim(sample_states(rng, buff, 1), SN+1, 1)
end

function sample_state(buff::ExperienceBuffer)
    return sample_state(Random.GLOBAL_RNG, buff)
end

function sample_experiences(rng::AbstractRNG, buff::ExperienceBuffer{ST, SN, AT, AN, R, SNp1, ANp1}, num_samples::Int; kwargs...) where {ST, SN, AT, AN, R, SNp1, ANp1}
    dest_states = Array{ST}(undef, size(buff.states)[1:end-1]..., num_samples)
    dest_actions = Array{AT}(undef, size(buff.actions)[1:end-1]..., num_samples)
    dest_rewards = Array{R}(undef, num_samples)
    dest_nextstates = Array{ST}(undef, size(buff.states)[1:end-1]..., num_samples)
    dest_nextstateisterminal = zeros(Bool, num_samples)
    dest_nextstateisabsorbing = zeros(Bool, num_samples)
    sample_experiences!(rng, buff, num_samples, dest_states, dest_actions, dest_rewards, dest_nextstates, dest_nextstateisterminal, dest_nextstateisabsorbing; kwargs...)
    return dest_states, dest_actions, dest_rewards, dest_nextstates, dest_nextstateisterminal, dest_nextstateisabsorbing
end


function sample_experiences(buff::ExperienceBuffer, args...; kwargs...)
    return sample_experiences(Random.GLOBAL_RNG, buff, args...; kwargs...)
end

"""In-place sampling"""
function sample_experiences!(rng::AbstractRNG, buff::ExperienceBuffer{ST, SN, AT, AN, R, SNp1, ANp1}, num_samples::Int, dest_states::AbstractArray{SP, SNp1}, dest_actions::Union{Nothing, AbstractArray{AP, ANp1}}=nothing, dest_rewards::Union{Nothing, AbstractVector{RP}}=nothing, dest_nextstates::Union{Nothing, AbstractArray{SP, SNp1}}=nothing, dest_nextstateisterminal::Union{Nothing, AbstractVector{Bool}}=nothing, dest_nextstateisabsorbing::Union{Nothing, AbstractVector{Bool}}=nothing; state_preprocess_fn=identity, action_preprocess_fn=identity, reward_preprocess_fn=identity, included_indices=:all) where {ST, SN, AT, AN, R, SNp1, ANp1, SP, AP, RP}

    @assert size(dest_states) == (size(buff.states)[1:end-1]..., num_samples)
    @assert dest_actions === nothing || size(dest_actions) == (size(buff.actions)[1:end-1]..., num_samples)
    @assert dest_rewards === nothing || length(dest_rewards) == num_samples
    @assert dest_nextstates === nothing || size(dest_nextstates) == (size(buff.states)[1:end-1]..., num_samples)
    @assert dest_nextstateisterminal === nothing || length(dest_nextstateisterminal) == num_samples
    @assert dest_nextstateisabsorbing === nothing || length(dest_nextstateisabsorbing) == num_samples

    if included_indices == :all; included_indices=1:length(buff); end
    j::Int = 1
    while (j <= size(dest_states)[end])
        si::Int = rand(rng, included_indices)
        (si == circularindex(buff._idx, buff.capacity)) && continue  # don't sample the latest state since the next state has not come in yet
        buff.isterminals[si]  && continue
        @assert si == circularindex(si, buff.capacity) "This shouldn't be possible. A state cannot be non-terminal but still absorbing"
        nsi::Int = circularindex(si + 1, buff.capacity)
        

        selectdim(dest_states, SN+1, j) .= state_preprocess_fn(selectdim(buff.states, SN+1, si))
        if dest_nextstates !== nothing
            selectdim(dest_nextstates, SN+1, j) .= state_preprocess_fn(selectdim(buff.states, SN+1, nsi))
        end
        if dest_actions !== nothing
            selectdim(dest_actions, AN+1, j) .= action_preprocess_fn(selectdim(buff.actions, AN+1, si))
        end
        if dest_rewards !== nothing
            dest_rewards[j] = reward_preprocess_fn(buff.rewards[si])
        end
        if dest_nextstateisterminal !== nothing
            dest_nextstateisterminal[j] = buff.isterminals[nsi]
        end
        if dest_nextstateisabsorbing !== nothing
            dest_nextstateisabsorbing[j] = buff.isabsorbing[nsi]
        end

        j+=1
    end

    return nothing
end


function sample_experiences!(buff::ExperienceBuffer, args...; kwargs...)
    return sample_experiences!(Random.GLOBAL_RNG, buff, args...; kwargs...)
end