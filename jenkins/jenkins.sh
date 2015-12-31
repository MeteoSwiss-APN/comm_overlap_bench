#!/bin/bash -f

### environment variables inherited from Jenkins
# slave              The name of the build slave (dom, opcode, todi, ...).
# target             The name of the target (cpu, gpu).
# BUILD_NUMBER       The current build number, such as "153".
# BUILD_ID           The current build id, such as "2005-08-22_23-59-59" (YYYY-MM-DD_hh-mm-ss).
# BUILD_DISPLAY_NAME The display name of the current build, something like "#153" by default.
# JOB_NAME           Name of the project of this build, such as "foo" or "foo/bar".
# BUILD_TAG          String of "jenkins-${JOB_NAME}-${BUILD_NUMBER}".
# EXECUTOR_NUMBER    The unique number that identifies the current executor.
# NODE_NAME          Name of the slave if the build is on a slave, or "master" if run on master.
# NODE_LABELS        Whitespace-separated list of labels that the node is assigned.
# WORKSPACE          The absolute path of the directory assigned to the build as a workspace.
# JENKINS_HOME       The absolute path of the data storage directory assigned on the master node.
# JENKINS_URL        Full URL of Jenkins, like http://server:port/jenkins/
# BUILD_URL          Full URL of this build, like http://server:port/jenkins/job/foo/15/
# JOB_URL            Full URL of this job, like http://server:port/jenkins/job/foo/
# SVN_REVISION       Subversion revision number that's currently checked out to the workspace.
# SVN_URL            Subversion URL that's currently checked out to the workspace.

exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

# echo basic setup
echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"

# start timer
T="$(date +%s)"

# check sanity of environment
test -n "$1" || exitError 1001 ${LINENO} "must pass an argument"
test -n "${BUILD_TAG}" || exitError 1002 ${LINENO} "BUILD_TAG is not defined"
test -n "${JOB_NAME}" || exitError 1003 ${LINENO} "JOB_NAME is not defined"
test -n "${WORKSPACE}" || exitError 1004 ${LINENO} "WORKSPACE is not defined"
test -n "${slave}" || exitError 1005 ${LINENO} "slave is not defined"
shortslave=`echo ${slave} | sed 's/[0-9]*$//g'`
echo "$1" | egrep '^build$|^test$' || exitError 1007 ${LINENO} "invalid argument (must be one of build, test)"

# check special modes
if [ -z "${JENKINS_LOCAL_MODE}" ] ; then
  export JENKINS_LOCAL_MODE=OFF
fi

# some global variables
action="$1"
optarg="$2"
library="dycore"

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
test -f ${envloc}/env/machineEnvironment.sh || exitError 1201 ${LINENO} "cannot find machineEnvironment.sh script"
. ${envloc}/env/machineEnvironment.sh

# check that host (define in machineEnvironment.sh) and slave are consistent
echo ${host} | grep "${shortslave}" || exitError 1006 ${LINENO} "host does not contain slave"

# act according to action specified
case "${action}" in

build ) 

    # check if build script exists
    script="./test/jenkins/build.sh"
    test -f "${script}" || exitError 1301 ${LINENO} "cannot find script ${script}"

    # check sane environment
    test -n "${build_type}" || exitError 1302 ${LINENO} "build_type is not defined"
    test -n "${real_type}" || exitError 1303 ${LINENO} "real_type is not defined"
    test -n "${target}" || exitError 1304 ${LINENO} "target is not defined"

    # run build script
    opt="-z -t ${target} -b"
    if [ "${JENKINS_LOCAL_MODE}" == "OFF" ] ; then
        if [ -z "${INSTALL_DIR}" ] ; then
            exitError 1305 ${LINENO} "INSTALL_DIR is not defined"
        fi
        opt="${opt} -i ${INSTALL_DIR}/${library}/${build_type}_${real_type}"
    else
        #opt="${opt} -l"   # not required since STELLA is not compiled locally
                           # but taken from INSTALL_DIR
        opt="${opt}"
    fi
    if [ ! -z "${build_type}" -a "${build_type}" == "debug" ] ; then
      opt="${opt} -d"
    fi
    if [ ! -z "${real_type}" -a "${real_type}" == "float" ] ; then
      opt="${opt} -4"
    fi
    ${script} ${optarg} ${opt}
    if [ $? -ne 0 ] ; then
      exitError 1305 ${LINENO} "problem while executing script ${script}"
    fi
    
    # check if required artefacts exist
    if [ "${target}" == "gpu" ] ; then
      suffix="CUDA"
    else
      suffix=""
    fi
    test -f ./build/src/dycore/libDycore${suffix}.a || exitError 1311 ${LINENO} "build artifact not found (build/src/dycore/libDycore${suffix}.a)"
    test -f ./build/src/wrapper/libDycoreWrapper${suffix}.a || exitError 1312 ${LINENO} "build artifact not found (build/src/wrapper/libDycoreWrapper${suffix}.a)"

    # success
    echo "BUILDING SUCESSFUL"

    ;;

test )

    # check if test script exists
    script="./test/jenkins/test.sh"
    test -f "${script}" || exitError 1401 ${LINENO} "cannot find script ${script}"
    
    # ensure we have the working precision defined
    test -n "${real_type}" || exitError 1411 ${LINENO} "real_type is not defined"

    # run test script
    opt=""
    if [ "${JENKINS_LOCAL_MODE}" == "ON" ] ; then
        opt="${opt} -l"
    fi
    ${script} ${optarg} ${opt}
    if [ $? -ne 0 ] ; then
      exitError 1413 ${LINENO} "problem while executing script ${script}"
    fi

    # success
    echo "TESTING SUCESSFUL"

    ;;

* )

    exitError 3001 ${LINENO} "unsupported action in $0 encountered ($action)"

    ;;

esac

# end timer and report time taken
T="$(($(date +%s)-T))"
printf "####### time taken: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"

# no errors encountered
echo "####### finished: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
exit 0

# so long, Earthling!
