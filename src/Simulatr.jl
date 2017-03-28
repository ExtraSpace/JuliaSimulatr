module Simulatr

using StatsBase
using Distributions
using NamedTuples

import StatsBase: sample
import Distributions: Uniform
import NamedTuples: @NT

export simrel

include("simrel.jl")


end # module
