using LinearAlgebra
using RRTStar

"""Computes whether or not each focus area contains part of the current best path."""
function contains_solution(rrt::RRTStarPlanner, path, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)
    contains_solution_scores = zeros(Float32, num_bins, num_bins)
    if rrt.costs[rrt.goal] == Inf
        return contains_solution_scores
    end
    for column in x_index_list
        for row in y_index_list
            for node in path
                x = node[1]
                y = node[2]
                if x_bin_lower_bounds[column] <= x && x < x_bin_upper_bounds[column] && y_bin_lower_bounds[row] <= y && y < y_bin_upper_bounds[row]
                    contains_solution_scores[row, column] = 1.0
                    break
                end
            end
        end
    end
    return contains_solution_scores
end


"""Computes fraction of solution nodes in each focus area"""
function fraction_of_solution_nodes(rrt::RRTStarPlanner, path, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)
    fractional_solution_scores = zeros(Float32, num_bins, num_bins)
    if rrt.costs[rrt.goal] == Inf
        return fractional_solution_scores
    end
    for column in x_index_list
        for row in y_index_list
            for node in path
                x = node[1]
                y = node[2]
                if x_bin_lower_bounds[column] <= x && x < x_bin_upper_bounds[column] && y_bin_lower_bounds[row] <= y && y < y_bin_upper_bounds[row]
                    fractional_solution_scores[row, column] += 1.0
                end
            end
        end
    end
    fractional_solution_scores = normalize(fractional_solution_scores)
    return fractional_solution_scores
end


"""Computes fraction of total tree nodes in each focus area"""
function fraction_of_tree_nodes(rrt::RRTStarPlanner, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)
    fractional_tree_node_scores = zeros(Float32, num_bins, num_bins)
    for column in x_index_list
        for row in y_index_list
            for i in 1:rrt.new_batch_size
                x, y = rrt.new_batch[:,i]
                if x_bin_lower_bounds[column] <= x && x < x_bin_upper_bounds[column] && y_bin_lower_bounds[row] <= y && y < y_bin_upper_bounds[row]
                    fractional_tree_node_scores[row, column] += 1.0
                end
            end
            for i in 1:rrt.num_total_points
                x, y = rrt.all_points[:,i]
                if x_bin_lower_bounds[column] <= x && x < x_bin_upper_bounds[column] && y_bin_lower_bounds[row] <= y && y < y_bin_upper_bounds[row]
                    fractional_tree_node_scores[row, column] += 1.0
                end
            end
        end
    end
    fractional_tree_node_scores = normalize(fractional_tree_node_scores .+ 0.001)
    return fractional_tree_node_scores
end


"""Computes curvature per node per cell"""
function cell_curvature(rrt::RRTStarPlanner, path, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)
    curvature_per_node_scores = zeros(Float32, num_bins, num_bins)
    cumulative_curvature = zeros(Float32, num_bins, num_bins)
    cumulative_node_counts = zeros(Float32, num_bins, num_bins)
    if rrt.costs[rrt.goal] == Inf
        return curvature_per_node_scores
    end
    for column in x_index_list
        for row in y_index_list
            for i in 2:(length(path)-1)
                node = path[i]
                x = node[1]
                y = node[2]
                if x_bin_lower_bounds[column] <= x && x < x_bin_upper_bounds[column] && y_bin_lower_bounds[row] <= y && y < y_bin_upper_bounds[row]
                    v_1 = collect(node) - collect(path[i-1])
                    v_2 = collect(path[i+1]) - collect(node)
                    v_1 = normalize(v_1)
                    v_2 = normalize(v_2)
                    curvature = 1 - dot(v_1, v_2)
                    cumulative_curvature[row, column] += curvature
                    cumulative_node_counts[row, column] += 1.0
                end
            end
        end
    end
    # Prevent division by zero
    curvature_per_node_scores = cumulative_curvature ./ (cumulative_node_counts .+ 0.001)
    return curvature_per_node_scores
end


"""Computes the fraction of samples made in each cell, after applying the steering function"""
function fraction_of_samples(rrt::RRTStarPlanner, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)
    fractional_sample_scores = zeros(Float32, num_bins, num_bins)
    for column in x_index_list
        for row in y_index_list
            for i in 1:rrt.num_points_sampled
                x, y = rrt.all_samples[:,i]
                if x_bin_lower_bounds[column] <= x && x < x_bin_upper_bounds[column] && y_bin_lower_bounds[row] <= y && y < y_bin_upper_bounds[row]
                    fractional_sample_scores[row, column] += 1.0
                end
            end
        end
    end
    
    # Prevent dicision by zero at a later step
    fractional_sample_scores = normalize(fractional_sample_scores .+ 0.001)
    return fractional_sample_scores
end


"""Computes the fraction of samples that have lead to a path cost improvement per cell"""
function fraction_of_improvements(rrt::RRTStarPlanner, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)
    fractional_improvement_scores = zeros(Float32, num_bins, num_bins)
    if rrt.costs[rrt.goal] == Inf
        return normalize(fractional_improvement_scores .+ 0.001)
    end
    for column in x_index_list
        for row in y_index_list
            for i in rrt.num_improving_points
                x, y = rrt.improving_points[:,i]
                if x_bin_lower_bounds[column] <= x && x < x_bin_upper_bounds[column] && y_bin_lower_bounds[row] <= y && y < y_bin_upper_bounds[row]
                    fractional_improvement_scores[row, column] += 1.0
                end
            end
        end
    end
    fractional_improvement_scores = normalize(fractional_improvement_scores)
    return fractional_improvement_scores
end


# NOTE: potential other features
"""Distance of cell center to closest solution line seg / node"""
function compute_focus_area_scores(rrt::RRTStarPlanner)
    x_size = rrt.x_range[2] - rrt.x_range[1]
    y_size = rrt.y_range[2] - rrt.y_range[1]
    num_bins = 4
    x_bin_size = x_size / num_bins
    y_bin_size = y_size / num_bins
    x_index_list = 1:num_bins
    y_index_list = 1:num_bins
    x_bin_centers = range(rrt.x_range[1] + (x_bin_size / 2), rrt.x_range[2] - (x_bin_size / 2), length=num_bins)
    y_bin_centers = range(rrt.y_range[1] + (y_bin_size / 2), rrt.y_range[2] - (y_bin_size / 2), length=num_bins)
    x_bin_upper_bounds = range(rrt.x_range[1] + x_bin_size, rrt.x_range[2], length=num_bins)
    y_bin_upper_bounds = range(rrt.y_range[1] + y_bin_size, rrt.y_range[2], length=num_bins)
    x_bin_lower_bounds = range(rrt.x_range[1], rrt.x_range[2] - x_bin_size, length=num_bins)
    y_bin_lower_bounds = range(rrt.y_range[1], rrt.y_range[2] - y_bin_size, length=num_bins)

    solution_so_far = extract_best_path_reversed(rrt)

    contains_solution_scores = contains_solution(rrt, solution_so_far, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)
    fractional_solution_scores = fraction_of_solution_nodes(rrt, solution_so_far, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)
    fractional_tree_coverage_scores = fraction_of_tree_nodes(rrt, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)
    curvature_per_node_scores = cell_curvature(rrt, solution_so_far, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)
    fractional_sample_score = fraction_of_samples(rrt, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)
    fractional_improvements_score = fraction_of_improvements(rrt, x_index_list, y_index_list, x_bin_upper_bounds, y_bin_upper_bounds, x_bin_lower_bounds, y_bin_lower_bounds, num_bins)

    probability_of_improving = normalize(fractional_improvements_score ./ fractional_sample_score)

    # rrt.focus_area_scores = (contains_solution_scores + fractional_solution_scores .-1)
    # rrt.focus_area_scores = contains_solution_scores
    # rrt.focus_area_scores = tanh.(fractional_solution_scores .- fractional_tree_coverage_scores)
    # rrt.focus_area_scores = tanh.(contains_solution_scores .+ fractional_solution_scores .- fractional_tree_coverage_scores .+ curvature_per_node_scores .-1)
    # rrt.focus_area_scores = tanh.(vcat(contains_solution_scores, fractional_solution_scores, fractional_tree_coverage_scores, curvature_per_node_scores))

    rrt.focus_area_scores = vcat(contains_solution_scores, fractional_solution_scores, fractional_tree_coverage_scores, curvature_per_node_scores, probability_of_improving)
    return rrt.focus_area_scores
end