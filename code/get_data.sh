#!/bin/bash

searchstring_c="Best energy"
alt_searchstring_c="Starting with instructions"
searchstring_nc="best instrucitons after the non-clifford opimizaton"

path_h2="h2/h2_bl_"
path_beh2="beh2/beh2_spaclust/beh2_wfn_bl_"
path_beh2p="beh2/beh2_permut/simulations/beh2_wfn_bl_"
path_n2="n2/n2_serial_bl_"
# echo "$searchstring"

# Load H2 data
echo "starting h2"
for b_l in 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 ; do  #<....put all bond lenghts here....>; do

    path="${path_h2}${b_l}/output1.txt"
    # Load Clifford instructions
    stuff_c=$(grep  -A 2 "$searchstring_c" $path | sed -n '2p' | sed 's,with instructions,,g' | sed 's,\] \[\[,\] |\[\[,g')
    gates_c=$(echo "$stuff_c" | cut -d '|' -f 1 | cut -d '[' -f 2 | cut -d ']' -f 1 | sed "s,',,g")
    positions_c=$(echo "$stuff_c" | cut -d '|' -f 2) 
    positions_c=$(echo "$positions_c" | sed 's,\[\[,\[,' | sed 's,\]\],\],g' | sed 's/\], \[/\];\[/g')
    echo "$gates_c" > "files/h2_clifford_gates_${b_l}.txt"
    echo "$positions_c" > "files/h2_clifford_positions_${b_l}.txt"

    # Load non-Clifford instructions
    stuff_nc=$(grep  -A 2 "$searchstring_nc" $path | tail -1 | sed 's/with instructions//g'  | sed 's,\] \[\[,\] |\[\[,g')
    # echo $stuff_nc
    # stuff2=$(grep  -A 3 "$searchstring" $path | tail -2 | cut -d "[" -f 2)
    # echo $stuff
    # this is "\t with instructions [...]" -> want only [...]
    # gates_nc=$(echo "$stuff_nc" | cut -d '|' -f 1)
    gates_nc=$(echo "$stuff_nc" | cut -d '|' -f 1 | cut -d '[' -f 2 | cut -d ']' -f 1 | sed "s,',,g")
    positions_nc=$(echo "$stuff_nc" | cut -d '|' -f 2)
    positions_nc=$(echo "$positions_nc" | sed 's,\[\[,\[,' | sed 's,\]\],\],g' | sed 's/\], \[/\];\[/g') # | sed "s/\), \(/\);\(/g") 
    echo "$gates_nc" >     "files/h2_nclifford_gates_${b_l}.txt"
    echo "$positions_nc" > "files/h2_nclifford_positions_${b_l}.txt"
done

# Load BeH2 data
echo "starting beh2"
for b_l in 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 ; do  #<....put all bond lenghts here....>; do
    # echo "Clifford instructions gates BeH2" >          "files/beh2_clifford_gates_${b_l}.txt"
    # echo "Clifford instructions positions BeH2" >      "files/beh2_clifford_positions_${b_l}.txt"
    # echo "Near-Clifford instructions gates BeH2" >     "files/beh2_nclifford_gates_${b_l}.txt"
    # echo "Near-Clifford instructions positions BeH2" > "files/beh2_nclifford_positions_${b_l}.txt"

    path="${path_beh2}${b_l}/output.txt"
    # Load Clifford instructions
    stuff_c=$(grep  -A 2 "$searchstring_c" $path | sed -n '2p' | sed 's,with instructions,,g' | sed 's,\] \[\[,\] |\[\[,g')
    gates_c=$(echo "$stuff_c" | cut -d '|' -f 1 | cut -d '[' -f 2 | cut -d ']' -f 1 | sed "s,',,g")
    positions_c=$(echo "$stuff_c" | cut -d '|' -f 2) 
    positions_c=$(echo "$positions_c" | sed 's,\[\[,\[,' | sed 's,\]\],\],g' | sed 's/\], \[/\];\[/g')
    echo "$gates_c" > "files/beh2_clifford_gates_${b_l}.txt"
    echo "$positions_c" > "files/beh2_clifford_positions_${b_l}.txt"

    # Load non-Clifford instructions
    stuff_nc=$(grep  -A 2 "$searchstring_nc" $path | tail -1 | sed 's/with instructions//g'  | sed 's,\] \[\[,\] |\[\[,g')
    # echo $stuff_nc
    # stuff2=$(grep  -A 3 "$searchstring" $path | tail -2 | cut -d "[" -f 2)
    # echo $stuff
    # this is "\t with instructions [...]" -> want only [...]
    # gates_nc=$(echo "$stuff_nc" | cut -d '|' -f 1)
    gates_nc=$(echo "$stuff_nc" | cut -d '|' -f 1 | cut -d '[' -f 2 | cut -d ']' -f 1 | sed "s,',,g")
    positions_nc=$(echo "$stuff_nc" | cut -d '|' -f 2)
    positions_nc=$(echo "$positions_nc" | sed 's,\[\[,\[,' | sed 's,\]\],\],g' | sed 's/\], \[/\];\[/g') # | sed "s/\), \(/\);\(/g") 
    echo "$gates_nc" >     "files/beh2_nclifford_gates_${b_l}.txt"
    echo "$positions_nc" > "files/beh2_nclifford_positions_${b_l}.txt"

    # echo "Clifford instructions gates BeH2 permut" >          "files/beh2permut_clifford_gates_${b_l}.txt"
    # echo "Clifford instructions positions BeH2 permut" >      "files/beh2permut_clifford_positions_${b_l}.txt"
    # echo "Near-Clifford instructions gates BeH2 permut" >     "files/beh2permut_nclifford_gates_${b_l}.txt"
    # echo "Near-Clifford instructions positions BeH2 permut" > "files/beh2permut_nclifford_positions_${b_l}.txt"

    path="${path_beh2p}${b_l}/output.txt"
    # Load Clifford instructions
    stuff_c=$(grep  -A 2 "$searchstring_c" $path | sed -n '2p' | sed 's,with instructions,,g' | sed 's,\] \[\[,\] |\[\[,g')
    gates_c=$(echo "$stuff_c" | cut -d '|' -f 1 | cut -d '[' -f 2 | cut -d ']' -f 1 | sed "s,',,g")
    positions_c=$(echo "$stuff_c" | cut -d '|' -f 2) 
    positions_c=$(echo "$positions_c" | sed 's,\[\[,\[,' | sed 's,\]\],\],g' | sed 's/\], \[/\];\[/g')
    echo "$gates_c" > "files/beh2permut_clifford_gates_${b_l}.txt"
    echo "$positions_c" > "files/beh2permut_clifford_positions_${b_l}.txt"

    # Load non-Clifford instructions
    stuff_nc=$(grep  -A 2 "$searchstring_nc" $path | tail -1 | sed 's/with instructions//g'  | sed 's,\] \[\[,\] |\[\[,g')
    # echo $stuff_nc
    # stuff2=$(grep  -A 3 "$searchstring" $path | tail -2 | cut -d "[" -f 2)
    # echo $stuff
    # this is "\t with instructions [...]" -> want only [...]
    gates_nc=$(echo "$stuff_nc" | cut -d '|' -f 1 | cut -d '[' -f 2 | cut -d ']' -f 1 | sed "s,',,g")
    #gates_nc=$(echo "$stuff_nc" | cut -d '|' -f 1)
    positions_nc=$(echo "$stuff_nc" | cut -d '|' -f 2)
    positions_nc=$(echo "$positions_nc" | sed 's,\[\[,\[,' | sed 's,\]\],\],g' | sed 's/\], \[/\];\[/g') # | sed "s/\), \(/\);\(/g") 
    echo "$gates_nc" >     "files/beh2permut_nclifford_gates_${b_l}.txt"
    echo "$positions_nc" > "files/beh2permut_nclifford_positions_${b_l}.txt"

done
echo "done with beh2"


# Load N2 data
echo "starting n2"
for b_l in 0.75 1.0 1.3 1.5 1.75 2.0 2.25 2.5 2.75 3.0 ; do  #<....put all bond lenghts here....>; do
    # echo "Clifford instructions gates N2" >          "files/n2_clifford_gates_${b_l}.txt"
    # echo "Clifford instructions positions N2" >      "files/n2_clifford_positions_${b_l}.txt"
    # echo "Near-Clifford instructions gates N2" >     "files/n2_nclifford_gates_${b_l}.txt"
    # echo "Near-Clifford instructions positions N2" > "files/n2_nclifford_positions_${b_l}.txt"

    path="${path_n2}${b_l}/output.txt"

    # Load Clifford instructions
    if grep "$alt_searchstring_c" $path; then
        stuff_c=$(grep  "$alt_searchstring_c" $path  | sed 's,Starting with instructions,,g' | sed 's,\] \[\[,\] |\[\[,g')
    else
        stuff_c=$(grep  -A 2 "$searchstring_c" $path | sed -n '2p' | sed 's,with instructions,,g' | sed 's,\] \[\[,\] |\[\[,g')
    fi
    gates_c=$(echo "$stuff_c" | cut -d '|' -f 1 | cut -d '[' -f 2 | cut -d ']' -f 1 | sed "s,',,g")
    positions_c=$(echo "$stuff_c" | cut -d '|' -f 2) 
    positions_c=$(echo "$positions_c" | sed 's,\[\[,\[,' | sed 's,\]\],\],g' | sed 's/\], \[/\];\[/g')
    echo "$gates_c" > "files/n2_clifford_gates_${b_l}.txt"
    echo "$positions_c" > "files/n2_clifford_positions_${b_l}.txt"

    # Load non-Clifford instructions
    stuff_nc=$(grep  -A 2 "$searchstring_nc" $path | tail -1 | sed 's/with instructions//g'  | sed 's,\] \[\[,\] |\[\[,g')
    # gates_nc=$(echo "$stuff_nc" | cut -d '|' -f 1)
    gates_nc=$(echo "$stuff_nc" | cut -d '|' -f 1 | cut -d '[' -f 2 | cut -d ']' -f 1 | sed "s,',,g")
    positions_nc=$(echo "$stuff_nc" | cut -d '|' -f 2)
    positions_nc=$(echo "$positions_nc" | sed 's,\[\[,\[,' | sed 's,\]\],\],g' | sed 's/\], \[/\];\[/g') # | sed "s/\), \(/\);\(/g") 
    echo "$gates_nc" >     "files/n2_nclifford_gates_${b_l}.txt"
    echo "$positions_nc" > "files/n2_nclifford_positions_${b_l}.txt"
done
echo "done with n2"

 
