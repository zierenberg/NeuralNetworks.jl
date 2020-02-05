#module Networks
using LightGraphs
using Random
using Distributions
using LinearAlgebra

"""
    orlandi_topology()

Generate a network topology 

- L   = 5mm
- rho = 400 neurons/mm^2

# Default parameters (internal lengthscale in mm)
- Rs   =   7.5 # [μm] fixed soma radius
- Rd   = 150.0 # [μm] avg. dendritic radius (Gauss)
- sd   =  20.0 # [μm] variance of dendritic radii (Gauss)
- sa   = 900.0 # [μm] variance of axonal length (sigma in Rayleight) 
- la   =  10.0 # [μm] axon segment length
- sphi =  15.0/360*2*numpy.pi # [rad] Gaussian biased random walk
- pc   =   0.5 # prob. of connectivity if axon crosses dendritic tree
"""
#TODO: visualize
function orlandi_topology(L::Float64, rho::Float64, seed::Int; Rs=7.5e-3, Rd=150.0e-3, sd=20.0e-3, sa=900.0e-3, la=10.0e-3, sphi=15.0/360.0*2.0*pi, pc=0.5, verbose=false)
  rng = MersenneTwister(seed);
  
  N = Int(rho*L*L)
  print("Generate Orlandi network topology with N=$(N) neurons on a $(L)x$(L) mm^2 square.\n... ")

  list_dendritic_radius = rand(rng, Normal(Rd,sd),N);
  list_axonal_length    = rand(rng, Rayleigh(sa), N);

  if verbose
    print("Randomly place neurons in 2D space (can take long if density is too high).\n... ")
  end
  list_position = distribute_discs_uniformly_2D(N, L, rng, radius=Rs)

  if verbose
    print("Create domain decomposition for quicker evaluation of axonal dentritic tree crossing.\n... ")
  end
  max_Rd = maximum(list_dendritic_radius)
  nd_box = floor(Int, L/max_Rd)
  ld = L/nd_box
  # real number of domains include boundary domains outside of box on each side
  nd = nd_box + 2
  domains = LightGraphs.SimpleGraphs.grid([nd,nd], periodic=false) 
  domain_neurons = [Int[] for i=1:nv(domains)];
  for i in 1:N
    push!(domain_neurons[domain_of(list_position[i], ld, nd, offset=ld)], i)
  end

  if verbose
    print("Axonal growth for each neuron that leads to potential connections if axon crosses another neurons' dendritic tree.\n... ")
  end
  @inline function check_connection!(topology::SimpleDiGraph{Int64}, pos_axon::Vector{Float64}, id::Int, domain::Int)
    for d in [domain, outneighbors(domains,domain)...] 
      for j in domain_neurons[d]
        if intersect(pos_axon, list_position[j], list_dendritic_radius[j])
          add_edge!(topology, id, j)
        end
      end
    end
  end
  @inline function grow_axon(pos_axon::Vector{Float64}, phi::Float64, id::Int, la::Float64)
    phi = rand(rng, Normal(phi,sphi))
    pos_axon += la*[cos(phi),sin(phi)]
    # axons can grow outside of domain in the current implementation
    if in_range(pos_axon, -ld, L+ld)
      domain = domain_of(pos_axon, ld, nd, offset=ld)
      check_connection!(topology, pos_axon, id, domain)
    end
    return pos_axon, phi
  end

  topology = SimpleDiGraph(N)
  for id in 1:N 
    num_seg   = floor(Int, list_axonal_length[id]/la)
    remainder = list_axonal_length[id] - num_seg*la
    #random initial direction at edge of soma (Rs)
    phi = rand(rng)*2*pi
    pos_axon = list_position[id] + Rs*[cos(phi),sin(phi)]
    for i in 1:num_seg 
      pos_axon, phi = grow_axon(pos_axon, phi, id, la)
    end
    #do remainder segment
    pos_axon, phi = grow_axon(pos_axon, phi, id, remainder)
  end

  if verbose
    print("Sparsen potential connections with probability pc=$(pc).\n... ")
  end
  for e in edges(topology)
    # remove self-connections
    if src(e) == dst(e) 
      rem_edge!(topology,e)
    else 
      # keep other edges with probability pc 
      if ! (rand(rng) < pc)
        rem_edge!(topology,e)
      end
    end
  end

  println("done")
  return topology, list_position
end

###############################################################################
###############################################################################
### Helper Functions
@inline function intersect(pos::Vector{Float64}, ref::Vector{Float64}, R::Float64)::Bool
  if LinearAlgebra.norm(pos-ref) < R 
    return true
  end
  return false
end

@inline function overlap(possible_position::Vector{Float64}, list_position::Vector{Vector{Float64}}, radius::Float64)::Bool
  for position in list_position
    if intersect(possible_position, position, 2*radius)
      return true
    end
  end
  return false
end

#TODO: optimize overlap function
function distribute_discs_uniformly_2D(N::Int,L::Float64,rng::AbstractRNG; radius::Float64=0.0)::Vector{Vector{Float64}}
  list_position = Vector{Float64}[]
  for i in 1:N
    while true
      possible_position = rand(rng,2)*L 
      if ! overlap(possible_position, list_position, radius)
        push!(list_position, possible_position)
        break
      end
    end
  end
  return list_position
end

@inline function domain_of(position::Vector{Float64}, ld::Float64, nd::Int; offset::Float64=0.0)::Int
  ix = 1 + floor(Int, (position[1]+offset)/ld)
  iy = 1 + floor(Int, (position[2]+offset)/ld)
  return (iy-1)*nd + ix
end


function apply_reflective_boundary_condition(pos::Vector{Float64}, phi::Float64, dim_min::Float64, dim_max::Float64)
  num_reflections = 0
  for d in 1:length(pos)
    if pos[d] < dmin_min
      pos[d] = dim_min + (dim_min-pos[d])
    end
    if pos[d] > dmin_max
      pos[d] = dim_max + (dim_max-pos[d])
    end
  end
  return pos, phi
end

function in_range(pos::Vector{Float64}, dim_min::Float64, dim_max::Float64)
  for d in 1:length(pos)
    if pos[d] < dim_min
      return false
    end
    if pos[d] > dim_max
      return false
    end
  end
  return true
end

#end
#export Networks
