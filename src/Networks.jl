#module Networks
using LightGraphs
using Random
using Distributions
using LinearAlgebra
using HDF5

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
function metric_correlations(L::Float64, rho::Float64, seed::Int; Rs=7.5e-3, Rd=150.0e-3, sd=20.0e-3, sa=900.0e-3, la=10.0e-3, sphi=15.0/360.0*2.0*pi, pc=0.5, verbose=false, write_to="")
  rng = MersenneTwister(seed);
  
  N = Int(rho*L*L)
  print("Generate network topology with metric correlations inspired by [Orlandi et al., Nat. Phys., 2013] with N=$(N) neurons on a $(L)x$(L) mm^2 square.\n... ")

  if verbose
    print("Uniformly distributed neurons in 2D space (can take long if density is too high).\n... ")
  end
  list_position_neuron = distribute_discs_uniformly_2D(N, L, rng, radius=Rs)

  if verbose
    print("Normally distributed dendritic tree radii with domain decomposition for quicker evaluation of axonal dentritic tree crossing.\n... ")
  end
  list_dendritic_radius = rand(rng, Normal(Rd,sd),N);
  system = system_2D_square(L, list_position_neuron, list_dendritic_radius)

  if verbose
    print("Axonal growth for each neuron (length follows Rayleigh distribution) that leads to potential connections if axon crosses another neurons' dendritic tree.\n... ")
  end
  list_axonal_length = rand(rng, Rayleigh(sa), N);
  if write_to!=""
    list_position_axon = Array{Float64,2}[]
  end
  topology = SimpleDiGraph(N)
  for id in 1:N 
    #grow axom from random initial direction at edge of soma (Rs)
    phi = rand(rng)*2*pi
    pos_axon = list_position_neuron[id] + Rs*[cos(phi),sin(phi)]
    position_axon = grow_axon(pos_axon, list_axonal_length[id], la, phi, sphi, rng)
    if write_to!=""
      push!(list_position_axon, position_axon)
    end
    
    add_edges!(topology, id, position_axon, system)
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

  if write_to!=""
    f5 = h5open(write_to, "w")
    group_position_neuron = g_create(f5,"position_neuron")
    group_position_axon   = g_create(f5,"position_axon")
    group_out_neighbors   = g_create(f5,"graph_out_neighbors")

    for i in 1:N
      group_position_neuron["$(i)"]= list_position_neuron[i]
      group_position_axon["$(i)"]= list_position_axon[i]
      group_out_neighbors["$(i)"]= outneighbors(topology,i)
    end
     
    close(f5)
  end

  println("done")
  return topology
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

function grow_axon(pos_start::Vector{Float64}, length_total::Float64, length_segment::Float64, phi_start::Float64, sphi::Float64, rng::AbstractRNG)
  num_seg   = floor(Int, length_total/length_segment)
  remainder = length_total - num_seg*length_segment
  list_position_axon = zeros(num_seg+2,2)

  #initial conditions
  phi = phi_start
  list_position_axon[1,:] = pos_start

  for seg in 1:num_seg 
    phi, dx = unit_step_2D_biased_direction(phi,sphi,rng)
    list_position_axon[seg+1,:] = list_position_axon[seg,:] + length_segment*dx
  end

  #do remainder segment
  phi, dx = unit_step_2D_biased_direction(phi,sphi,rng)
  list_position_axon[num_seg+2,:] = list_position_axon[num_seg+1,:] + remainder*dx

  return list_position_axon
end

function unit_step_2D_biased_direction(phi::Float64, sphi::Float64, rng::AbstractRNG)
  phi = rand(rng, Normal(phi,sphi))
  dx  = [cos(phi),sin(phi)]
  return phi, dx
end

struct BoxSystem{F}
  dims::Vector{Float64}
  list_position_neuron::Vector{Vector{Float64}}
  list_dendritic_radius::Vector{Float64}
  in_range::F
  dims_domain::Vector{Float64}
  num_domains::Vector{Int}
  domains::SimpleGraph{Int64}
  domain_neurons::Vector{Vector{Int}}
end

function system_2D_square(L::Float64, list_position_neuron::Vector{Vector{Float64}}, list_dendritic_radius::Vector{Float64})::BoxSystem
  max_Rd = maximum(list_dendritic_radius)
  nd_box = floor(Int, L/max_Rd)
  ld = L/nd_box
  # real number of domains include boundary domains outside of box on each side
  nd = nd_box + 2
  domains = LightGraphs.SimpleGraphs.grid([nd,nd], periodic=false) 
  domain_neurons = [Int[] for i=1:nv(domains)];
  for i in 1:length(list_position_neuron)
    push!(domain_neurons[domain_of(list_position_neuron[i], ld, nd, offset=ld)], i)
  end
  function in_range(pos::Vector{Float64})
    for d in 1:length(pos)
      if pos[d] < -ld
        return false
      end
      if pos[d] > L+ld
        return false
      end
    end
    return true
  end
  system = BoxSystem([L,L], list_position_neuron, list_dendritic_radius, in_range, [ld,ld], [nd,nd], domains, domain_neurons)
  return system
end

function add_edges!(topology::SimpleDiGraph{Int64}, id::Int, list_pos_axon::Array{Float64,2}, system::BoxSystem)
  for i in 1:size(list_pos_axon,1)
    pos_axon = list_pos_axon[i,:]
    if system.in_range(pos_axon)
      domain = domain_of(pos_axon, system)
      for d in [domain, outneighbors(system.domains,domain)...] 
        for j in system.domain_neurons[d]
          if intersect(pos_axon, system.list_position_neuron[j], system.list_dendritic_radius[j])
            add_edge!(topology, id, j)
          end
        end
      end
    end
  end
end

@inline function domain_of(position::Vector{Float64}, system::BoxSystem)::Int
  index = 1
  scaling_factor = 1
  for i in 1:length(position)
    d_index_i = floor(Int, (position[i] + system.dims_domain[i])/system.dims_domain[i])
    index += d_index_i*scaling_factor
    scaling_factor *= system.num_domains[i]
  end
  return index
end

#end
#export Networks
