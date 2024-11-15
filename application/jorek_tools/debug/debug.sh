export OMP_NUM_THREADS=1
srun --cpu-bind=cores -A ukaea-ap002-cpu -p icelake-himem --mem=8000 -N 1 -n 1 -t 00:10:00 mpirun -n 1 ./jorek_model < $1
