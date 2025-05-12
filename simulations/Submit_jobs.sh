##!/bin/bash
##SBATCH -p amd_512
##SBATCH -N 2
#source /public3/soft/modules/module.sh
#module load mpi/openmpi/3.1.3-icc18-cjj-public3
#export PATH=/public3/home/scb9619/specfem3d-devel/bin:$PATH
#
#export PYTHONUNBUFFERED=1
#source /public3/soft/modules/module.sh
#module load miniforge gcc/9.3.0
#source activate py38

ulimit -s unlimited

SLIPMODEL="Myanmar_slip.dat"
MESHDIR="./MESH-Myanmar-uniform"
OUTPUTDIR="/public3/home/scb9619/3D_Myanmar/output"
NPROC=128
NSTEP=2000
DT=0.005
mud=0.4
S3=100e6
mu=30e9
model_name="Myanmar"
Lon_ref=96.0442
Lat_ref=21.9924
Gc_type=1   # 0: uniform Dc and S ratio; 1: variable Gc, dc indicate the energy ratio

for dc in 0.5
do
for S in 1.0
do

# cleans output files
mkdir -p OUTPUT_FILES
rm -rf OUTPUT_FILES/*

# stores setup
cp -r DATA/ OUTPUT_FILES/

BASEMPIDIR=`grep ^LOCAL_PATH DATA/Par_file | cut -d = -f 2 `
mkdir -p $BASEMPIDIR

# Revise the simulation parameters
gawk '{if($1=="NPROC")                         print $1,$2,"'"$NPROC"'";\
  else if($1=="NSTEP")                         print $1,$2,"'"$NSTEP"'";\
  else if($1=="DT")                            print $1,$2,"'"$DT"'"; \
  else print $0}' Template/Par_file > DATA/Par_file
# Revise the fault parameters
#sed  -e 's/key_dc/'$dc'/g'  ./Template/Par_file_faults > DATA/Par_file_faults
cp  ./Template/Par_file_faults  DATA/Par_file_faults

echo ${SLIPMODEL} ${mud} ${S3} ${S} ${dc} ${mu} ${Lon_ref} ${Lat_ref} ${Gc_type}
python calculate_stress.py -n ${SLIPMODEL} ${mud} ${S3} ${S} ${dc} ${mu} ${Lon_ref} ${Lat_ref} ${Gc_type}

### Run the simulation
# decomposes mesh
./bin/xdecompose_mesh $NPROC $MESHDIR $BASEMPIDIR
## runs database generation
mpirun -np $NPROC ./bin/xgenerate_databases
# runs simulation
mpirun -np $NPROC ./bin/xspecfem3D

# Save the output data
rm -rf ${OUTPUTDIR}/${model_name}_dc_${dc}_S_${S}
rm OUTPUT_FILES/DATABASES_MPI -r
mv OUTPUT_FILES ${OUTPUTDIR}/${model_name}_dc_${dc}_S_${S}

done
done
