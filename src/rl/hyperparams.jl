using Random
using IntervalSets
using ReinforcementLearning

export RLHyperparameters, to_dict

mutable struct RLHyperparameters
    env_seed::Int                       # To seed envrionment
    env_fixedhorizon_steps::Int         # Numer of steps at which the environment cuts off without transitioning to an absorbing state. Also known as "Timelimit" in OpenAI gym's nomenclature.

    logs_dir::String                    # Log directory. "logs/" by default.
    
    max_steps::Int                      # Run RL for these many max steps.
    max_episodes::Int                   # Run RL for these many max episodes.

    # Exploration parameters:
    exploration_seed::Int               # To seed exploration randomness
    min_explore_steps::Int              # Act randomly initially for these many steps
    ϵ_anneal_period::Int                # Linearly anneal ϵ from 1->ϵ over these many steps after initial random exploration
    ϵ::Float32                          # ϵ-greedy

    # TD-Update parameters:
    γ::Float32                          # Gamma for bellman updates. Normaly 0.99.
    n::Int                              # n-step TD update. By default n=1 (i.e., pure TD).
    sarsa::Bool                         # Learn ep-greedy policy instead of greedy. False by default.
    η::Float32                          # learning rate

    # Deep learning parameters
    learning_seed::Int                  # To seed neural network, minibatch sampling etc.
    experience_buffer_size::Int         # Capacity of circular experience buffer.
    minibatch_size::Int                 # Minibatch size for SGD updates.
    train_interval::Int                 # ratio of data to sgd_steps
    grad_clipping::Float32              # by norm
    ρ::Float32                          # Target-network update polyak
    Δτ::Int                             # Target-network update every these many sgd_steps
    
    hidden_dims::Tuple{Vararg{Int}}     # Dimensions of hiidden layers in the neural network
    σ::Function                         # activation function
    layernorm::Bool                     # use layer norm ?
    device::Function                    # gpu / cpu

    function RLHyperparameters()
        return new(0, typemax(Int), "logs/",
                1000000, 25000,
                0, 25000, 25000, 0.1,
                0.99, 1, false, 1e-3,
                0, 1000000, 32, 1, Inf32, 0.995, 1,
                (64, 32), relu, false, cpu)
    end
end

function to_dict(h::RLHyperparameters)
    return Dict(key => string(getfield(h, key)) for key in fieldnames(RLHyperparameters))
end

function eval_mode!(h::RLHyperparameters)
    h.min_explore_steps = 0
    h.ϵ_anneal_period = 0
    h.ϵ = 0
    h.α =0
    h.train_interval = typemax(Int)
end

