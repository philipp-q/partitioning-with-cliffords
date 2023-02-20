#! /bin/bash
export OMP_NUM_THREADS=1
echo "Present working Directory is: `pwd`"

# for b_l in 1.0 1.20 1.40 1.5 1.6 1.8 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 ; do
# for b_l in 2.0 2.2 2.4 2.6 2.8 3.0  ; do # beh2
# for b_l in 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 ; do # beh2
for b_l in 2.0  ; do # beh2
# --> use this for b_l in 1.0 1.20 1.40 1.5 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 4.0 ; do
 # dir="beh2-stuff/beh2_cliff_run/beh2_trash2_bl_${b_l}"
 dir="simulations/beh2_wfn_bl_${b_l}"
  #old_dir="production_run_Jan14/beh2_bl_${b_l}"
  mkdir $dir
  # cat main_heisenberg.py | sed "s|xxxx|False|; s|yyyy|True|; s|zzzz|False|; s|rrrr|$b_l|; s|ryry|$b_l|" > $dir/main.py
  cat ../../main_heisenberg.py | sed "s|xxxx|True|; s|yyyy|False|; s|zzzz|False|; s|rrrr|$b_l|;  s|bbbb|$b_l|; s|ryry|$b_l|" > $dir/main.py
  # cp ../../do_annealing.py  $dir/
  cp ../../grad_hacked.py $dir/
  # cp ../../energy_optimization.py $dir/
  cp ../../hacked* $dir/
  cp ../../HEA.py $dir/
  cp ../../parallel_annealing.py $dir/
  cp ../../mutation_options.py $dir/
  cp ../../my_mpo.py $dir/
  cp ../../scipy_optimizer.py $dir/
  cp ../../vqe_utils.py $dir/
  cp ../../single_thread_annealing.py $dir/
  #cp $old_dir/instruct_* $dir/
  cd $dir
  echo "Present working Directory is: `pwd`"
  # python main.py > output
  python main.py | tee output5.txt
  # cp instruct_* ../../
  cd ../../
  killall python
done
