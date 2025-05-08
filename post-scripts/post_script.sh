##!/bin/bash
##SBATCH -p amd_256
##SBATCH -N 1
##SBATCH -n 1
#export PYTHONUNBUFFERED=1
#source /public3/soft/modules/module.sh
#module load miniforge
#source activate py38

Model_dir="/public3/home/scb9619/3D_Myanmar/output"


for file in `ls -d ${Model_dir}/*`
do
model=`echo $file | gawk 'BEGIN{FS="output/"}{print $2}'`
if [ -f data/${model}-results.dat ] ;  then
#if [ ! -d ${Model_dir}/${model} ] ;  then
  continue
fi

echo "Post processing" $model 
python Present_specfem3D_results.py -n ${Model_dir} ${model} > temp-fault-$model 
echo "finish process"

done

wait

rm  temp-*
