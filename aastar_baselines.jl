using AnytimeWeightedAStar
using AnytimeWeightedAStar.SearchProblem
using AnytimeWeightedAStar.ExampleProblems
using Random
using DataFrames
using CSV

time_dependent_utility(q, t, α, β) = α*q - exp(β*t) + 1

function solve_all(search_problem, seed, num_instances, awa_weights, awa_dec_weights, nodes_budget, alpha, beta, filename)
    dict = Dict{String,Vector{Float64}}()
    dict["instance_id"] = []
    dict["h0"] = []
    Random.seed!(search_problem, seed)
    for instance_id in 1:num_instances
        reset!(search_problem)
        h0 = heuristic(search_problem, start_state(search_problem))
        push!(dict["instance_id"], instance_id)
        push!(dict["h0"], h0)
        for w in ["dec_$(awa_dec_weights)", awa_weights...]
            if w == "dec_$(awa_dec_weights)"
                isempty(awa_dec_weights) && continue
                awa = awastar_search_with_scheduled_weights(search_problem, awa_dec_weights, Inf, nodes_budget)
            else
                awa = awastar_search(search_problem, w, Inf, nodes_budget)
            end
            solution_quality = quality(awa.solution_cost, h0)
            utility = time_dependent_utility(solution_quality, 1, alpha, beta)
            isopt = awa.nodes_expended < nodes_budget
            @info "Solved instance" instance_id h0 w awa.nodes_expended awa.num_solutions awa.solution_cost solution_quality utility isopt SearchProblem.info(search_problem)...
            

            q_key = "q_$(string(w))"
            u_key = "u_$(string(w))"
            c_key = "c_$(string(w))"
            isopt_key = "isopt_$(string(w))"
            if !haskey(dict, q_key)
                dict[q_key] = Float64[]
                dict[u_key] = Float64[]
                dict[c_key] = Float64[]
                dict[isopt_key] = Float64[]
            end
            push!(dict[q_key], solution_quality)
            push!(dict[c_key], awa.solution_cost)
            push!(dict[u_key], utility)
            push!(dict[isopt_key], isopt)
        end
        instance_id % 10 == 0  &&  CSV.write(filename, DataFrame(dict))
    end

    CSV.write(filename, DataFrame(dict))
    
    return dict
end

function get_avg_solution_quality_per_approach(solution_qualities_per_approach)
    avg_per_approach = Dict{Any,Float64}()
    for approach in keys(solution_qualities_per_approach)
        avg_per_approach[approach] = sum(solution_qualities_per_approach[approach]) / length(solution_qualities_per_approach[approach])
    end
    return avg_per_approach
end


awa_weights = [1, 1.5, 2, 3, 4, 5]      # Run static awa* with each of these weights
awa_dec_weights = [5, 4, 3, 2, 1.5, 1]  # Run awa* with decreasing weights every time a solution is found.
num_instances = 1000
seed = 42
alpha = 1
beta = log(1 + 0.25)


search_problems = [
    (name="SlidingPuzzle", problem=SlidingPuzzle(4:4, 50:60), budget=100000),
    (name="InverseSlidingPuzzle", problem=SlidingPuzzle(4:4, 50:60, inverse=true), budget=100000), 
    (name="TravellingSalesmanProblem", problem=TSP(25:35, (0.0, 0.9)), budget=50000),
    (name="GridNavigationProblem", problem=GNP(1000:1000, (0.05, 0.1)), budget=500000)
]

Threads.@threads for sp in search_problems
    mkpath("logs/AWAStarMetareasoning/Baselines-$(sp.name)")
    qualities_per_approach = solve_all(sp.problem, seed, num_instances, awa_weights, awa_dec_weights,  sp.budget, alpha, beta, "logs/AWAStarMetareasoning/Baselines-$(sp.name)/data.csv")
    # avg_quality_per_approach = get_avg_solution_quality_per_approach(qualities_per_approach)
    # println("Average qualities:\n", avg_quality_per_approach)
end
