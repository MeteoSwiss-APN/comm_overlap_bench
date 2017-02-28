module load daint-gpu
old_prgenv=`module list -t 2>&1 | grep 'PrgEnv-'`
if [ -z "${old_prgenv}" ] ; then
    module load PrgEnv-gnu
else
    module swap ${old_prgenv} PrgEnv-gnu
fi

module load craype-accel-nvidia60
#module load Boost/1.63.0-CrayGNU-2016.11-Python-3.5.2

