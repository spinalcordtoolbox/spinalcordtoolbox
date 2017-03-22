# Schedule file optimized for slice-wise Tx,Ty co-registration
# Author: Julien Cohen-Adad

# 8mm scale
setscale 8
setoption smoothing 8
setoption paramsubset 3  0 0 0 1 0 0 0 0 0 0 0 0  0 0 0 0 1 0 0 0 0 0 0 0    0 0 1 0 0 0 0 0 0 0 0 0
clear U
clear UA
setrow UA 1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1
optimise 12 UA:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4 

# 4mm scale
setscale 4
setoption smoothing 4
setoption paramsubset 3  0 0 0 1 0 0 0 0 0 0 0 0  0 0 0 0 1 0 0 0 0 0 0 0    0 0 1 0 0 0 0 0 0 0 0 0
clear UB
clear UL
clear UM
# remeasure costs at this scale
clear U
measurecost 12 UA 0 0 0 0 0 0 rel
sort U
copy U UL
# optimise best 3 candidates
clear U
optimise 12 UL:1-3  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4
# also try the identity transform as a starting point at this resolution
clear UQ
setrow UQ  1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1
optimise 7 UQ  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4

copy U UM
# select best 4 optimised solutions and try perturbations of these
clear U
copy UM:1-4 U
optimise 12 UM:1-4  0.0   0.0   0.0   0.1   0.0   0.0   0.0  abs 4
optimise 12 UM:1-4  0.0   0.0   0.0  -0.1   0.0   0.0   0.0  abs 4
optimise 12 UM:1-4  0.0   0.0   0.0   0.0   0.1   0.0   0.0  abs 4
optimise 12 UM:1-4  0.0   0.0   0.0   0.0  -0.1   0.0   0.0  abs 4
sort U
clear UB
copy U UB

# 2mm scale
setscale 2
setoption smoothing 2
setoption paramsubset 3  0 0 0 1 0 0 0 0 0 0 0 0  0 0 0 0 1 0 0 0 0 0 0 0    0 0 1 0 0 0 0 0 0 0 0 0
clear U
clear UC
clear UD
clear UE
clear UF
# remeasure costs at this scale
measurecost 12 UB 0 0 0 0 0 0 rel
sort U
copy U UC
clear U
optimise 12  UC:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4
copy U UD
setoption boundguess 1
if MAXDOF > 7
 clear U
if MAXDOF > 7
 optimise 9  UD:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 1
copy U UE
if MAXDOF > 9
 clear U
if MAXDOF > 9
 optimise 12 UE:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 2
sort U
copy U UF

# 1mm scale
setscale 1
setoption smoothing 1
setoption boundguess 1
setoption paramsubset 3  0 0 0 1 0 0 0 0 0 0 0 0  0 0 0 0 1 0 0 0 0 0 0 0    0 0 1 0 0 0 0 0 0 0 0 0
clear U
# also try the identity transform as a starting point at this resolution
setrow UF  1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1
optimise 12 UF:1-2  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 1
sort U
