module Topology
using LightGraphs
using Random
using Distributions
using LinearAlgebra
using HDF5

#using ComplexNetworks

#TODO: import from ComplexNetworks.jl e.g. hierarchical etc
#import ComplexNetworks.modular
#import ComplexNetworks.hierarchical
 
include("topologies/metric_correlations.jl")

"""
    all_to_all(N::Int)

Simple all-to-all (mean-field) topology excluding self-connections if self_connected=false
"""
function all_to_all(N::Int; self_connected::Bool=true)::SimpleGraph{Int}
  topology = LightGraphs.complete_graph(N)

  if self_connected
    for i in 1:N
       add_edge!(topology, i, i)
    end
  end

  return topology
end

"""
Random fixed-indegree network 
"""


end
export Topology
