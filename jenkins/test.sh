#!/bin/bash -f

exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

showUsage()
{
    echo "usage: `basename $0` [-l] [-h]"
    echo ""
    echo "optional arguments:"
    echo "-h           show this help message and exit"
    echo "-l           local mode (don't access INSTALL_DIR)"
}

parseOptions()
{
    # set defaults
    localmode=OFF

    # process command line options
    while getopts ":hl" opt
    do
        case $opt in
        h) showUsage; exit 0 ;;
        l) localmode=ON ;;
        \?) showUsage; exitError 301 ${LINENO} "invalid command line option (-${OPTARG})" ;;
        :) showUsage; exitError 302 ${LINENO} "command line option (-${OPTARG}) requires argument" ;;
        esac
    done

}

##################################################

# echo basic setup
echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"

# start timer
T="$(date +%s)"

# parse command line options (pass all of them to function)
parseOptions $*

# TODO: HACK for Piz Kesch (currently testing is not supported)
if [ "${host}" == "kesch" ] ; then
  echo "*****************************************************************************************"
  echo -e "WARNING: Skippintg tests in $0 on line ${LINENO}"
  echo "*****************************************************************************************"
  exit 0
fi

# some global variables
maxsleep=7200
local_data="$(pwd)/../cosmo/test/serialize/data"

# workaround for todi
if [ "${slave}" == "todi" ] ; then
  . /etc/bash.bashrc
  . /opt/modules/default/init/bash
  module load subversion/1.6.17
fi

# check presence of env directory
pushd `dirname $0` > /dev/null
envloc=`/bin/pwd`
popd > /dev/null
if [ ! -d ${envloc}/env ] ; then
  cd ${envloc}
  svn co svn+ssh://scm.hpcforge.org/var/lib/gforge/chroot/scmrepos/svn/cclm-dev/trunk/env
  if [ $? -ne 0 ] ; then
    exitError 1101 ${LINENO} "problem checking out environment files"
  fi
  cd -
else
  cd ${envloc}/env
  svn update
  if [ $? -ne 0 ] ; then
    exitError 1102 ${LINENO} "problem updating environment files"
  fi
  cd -
fi

# setup module environment and default queue
if [ ! -f ${envloc}/env/machineEnvironment.sh ] ; then
    exitError 1201 ${LINENO} "could not find ${envloc}/env/machineEnvironment.sh"
fi
. ${envloc}/env/machineEnvironment.sh

# load machine dependent functions
if [ ! -f ${envloc}/env/env.${host}.sh ] ; then
    exitError 1202 ${LINENO} "could not find ${envloc}/env/env.${host}.sh"
fi
. ${envloc}/env/env.${host}.sh

# load slurm tools
if [ ! -f ${envloc}/env/slurmTools.sh ] ; then
    exitError 1203 ${LINENO} "could not find ${envloc}/env/slurmTools.sh"
fi
. ${envloc}/env/slurmTools.sh

# go to build dir (if not already there)
if [ -d build ] ; then
  cd build
fi

# setup environment for test (same as dycore)
script="../modules_cpp.env"
test -f ${script} || exitError 6432 ${LINENO} "cannot find file ../modules_cpp.env"
prgenv=`module list -t 2>&1 | grep 'PrgEnv-'`
if [ -n "${prgenv}" ] ; then
    module unload ${prgenv}
fi
echo "source ${script}"
source ${script}

# make sure testdata exists
if [ "${localmode}" == "ON" ] ; then
    testdatadir="${local_data}"
else
    test -n "${TESTDATA_DIR}" || exitError 6434 ${LINENO} "TESTDATA_DIR is not defined"
    test -n "${real_type}" || exitError 6434 ${LINENO} "real_type is not defined"
    test -d "${TESTDATA_DIR}" || exitError 6436 ${LINENO} "${TESTDATA_DIR} can not be accessed"
    test -d "${TESTDATA_DIR}/${real_type}" || exitError 6438 ${LINENO} "${TESTDATA_DIR}/${real_type} can not be accessed"
    testdatadir="${TESTDATA_DIR}/${real_type}"
fi
echo ">>> Using testdata from: ${testdatadir}"
ntests=`find "${testdatadir}/." -maxdepth 1 -mindepth 1 -type d ! -iname ".svn" | wc -l` 
if [ "${ntests}" -le 0 ] ; then
  exitError 6439 ${LINENO} "no test found in ${testdatadir}"
fi

# setup
name="jenkins"
script="${envloc}/env/submit.${host}.slurm"
test -f ${script} || exitError 6440 ${LINENO} "cannot find script ${script}"

# loop over test data directories (and launch in parallel)
for testdata in `find "${testdatadir}/." -maxdepth 1 -mindepth 1 -type d ! -iname ".svn"` ; do

  # setup
  testname=`basename ${testdata}`
  out="${name}.${testname}.out"
  slurm="${name}.${testname}.slurm"

  # generate command
  cmd="export DYCORE_TESTDATA=${testdata}; ctest -VV"

  # run command (with SLURM job or interactively)
  /bin/cp -f ${script} ${slurm} || exitError 6444 ${LINENO} "problem copying batch job"
  /bin/sed -i 's|<NAME>|'"${name}"'|g' ${slurm}
  /bin/sed -i 's|<NTASKS>|1|g' ${slurm}
  /bin/sed -i 's|<NTASKSPERNODE>|1|g' ${slurm}
  if [ "${target}" == "gpu" ] ; then
    ## GPU
    # We expect the job to run with one GPU
    /bin/sed -i 's|<CPUSPERTASK>|'"1"'|g' ${slurm}
  else
    # CPU 
    # Test OpenMP 
    /bin/sed -i 's|<CPUSPERTASK>|'"${nthreads}"'|g' ${slurm}
  fi
  /bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${slurm}
  /bin/sed -i 's|<OUTFILE>|'"${out}"'|g' ${slurm}
  launch_job ${slurm} ${maxsleep} &

done

# wait for all jobs to finish
wait

# check if everything ran ok
for testdata in `find "${testdatadir}/." -maxdepth 1 -mindepth 1 -type d ! -iname ".svn"` ; do

  # setup
  testname=`basename ${testdata}`
  out="${name}.${testname}.out"

  # check if generation has been successfull
  egrep -i '^100% tests passed, 0 tests failed out of ' ${out}
  if [ $? -ne 0 ] ; then
    # echo output to stdout
    test -f ${out} || exitError 6550 ${LINENO} "batch job output file missing"
    echo "=== ${out} BEGIN ==="
    cat ${out} | /bin/sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g"
    echo "=== ${out} END ==="
    # abort
    exitError 4654 ${LINENO} "problem with unittests for test data ${testname} detected"
  else
    echo "Unittests for ${testname} successfull (see ${out} for detailed log)"
  fi

done

# end timer and report time taken
T="$(($(date +%s)-T))"
printf "####### time taken: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"

# no errors encountered
echo "####### finished: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
exit 0

# so long, Earthling!
