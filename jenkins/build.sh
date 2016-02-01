#!/bin/bash -f

# set some globals
library="hori_diff_tests"

##################################################
# functions
##################################################

exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

showUsage()
{
    echo "usage: `basename $0` [-4] [-c compiler] [-d] [-e] [-h] [-i path] [-t target] [-z]"
    echo ""
    echo "optional arguments:"
    echo "-4           build single precision version"
    echo "-b           be nice (build with low priority)"
    echo "-c compiler  compiler to use (e.g. cray)"
    echo "-d           debug mode on"
    echo "-e           write module environments .env files and exit"
    echo "-h           show this help message and exit"
    echo "-i path      override default install path"
    echo "-l           local mode"
    echo "-t target    build target (e.g. gpu)"
    echo "-s stelladir Directory where STELLA is installed"
    echo "-z           do a clean build"
}

parseOptions()
{
    # set defaults
    cleanup=OFF
    debug=OFF
    single=OFF
    writeenv=OFF
    installpath=""
    stelladir=""
    localmode=OFF
    nice_cmd=""
    nthreads=4

    # process command line options
    while getopts ":4bc:dehi:ls:t:z" opt
    do
        case $opt in
        4) single=ON ;;
        b) nice_cmd="nice" ;;
        c) compiler=$OPTARG ;;
        d) debug=ON ;;
        e) writeenv=ON ;;
        h) showUsage; exit 0 ;;
        i) installpath=$OPTARG ;;
        l) localmode=ON ;;
        s) stelladir=$OPTARG ;;
        t) target=$OPTARG ;;
        z) cleanup=ON ;;
        \?) showUsage; exitError 301 ${LINENO} "invalid command line option (-${OPTARG})" ;;
        :) showUsage; exitError 302 ${LINENO} "command line option (-${OPTARG}) requires argument" ;;
        esac
    done

    # check that everything is set
    test -n "${compiler}" || exitError 303 ${LINENO} "Option <compiler> is not set"
    test -n "${debug}" || exitError 304 ${LINENO} "Option <debug> is not set"
    test -n "${single}" || exitError 304 ${LINENO} "Option <single> is not set"
    test -n "${writeenv}" || exitError 305 ${LINENO} "Option <writeenv> is not set"
    test -n "${target}" || exitError 306 ${LINENO} "Option <taret> is not set"
    test -n "${cleanup}" || exitError 307 ${LINENO} "Option <debug> is not set"
    test -n "${MAKE_BUILD_THREADS}" && nthreads=${MAKE_BUILD_THREADS}

    # check for valid target
    containsElement "${target}" "${targets[@]}" || exitError 311 ${LINENO} "Invalid target (${target}) chosen"

    # check for valid compilers
    containsElement "${compiler}" "${compilers[@]}" || exitError 317 ${LINENO} "Invalid compiler (${compiler}) chosen"

    # check for localmode and STELLA dir
    if [ "${localmode}" == "ON" ] ; then
        if [ -n "${stelladir}" ] ; then
            echo "WARNING: specified STELLA_DIR with -s option has precendence over local mode (-l)"
        else
            stelladir="$(readlink -f '../stella/install')"
            test -n "${stelladir}" || exitError 319 ${LINENO} "Local STELLA directory does not exist (${stelladir})"
        fi
    fi
    if [ -n "${stelladir}" ] ; then
        test -d "${stelladir}" || exitError 318 ${LINENO} "Specified STELLA directory does not exist (${stelladir})"
    fi
}


showBuildConfiguration()
{

    # check whether SVN is available and this is a checkout
    which svn &> /dev/null
    if [ $? -eq 0 ] ; then
        svn info &> /dev/null
        if [ $? -eq 0 ] ; then
            local svn_path=`svn info | grep '^URL' | awk '{print $2}'`
            local svn_rev=`svn info | grep '^Revision' | awk '{print $2}'`
        fi
    fi

    # echo configuration to stdout
    echo "============================================================================="
    echo "build ${library}"
    echo "============================================================================="
    echo "date              : " `date`
    echo "machine           : " ${host}
    echo "user              : " `whoami`
    if [ -n "${svn_rev}" ] ; then
        echo "SVN path          : " ${svn_path}
        echo "SVN revision      : " ${svn_rev}
    else
        echo "SVN path          : " "(not available)"
        echo "SVN revision      : " "(not available)"
    fi
    echo "working in        : " ${base_path}
    echo "target            : " ${target}
    echo "compiler          : " ${compiler}
    echo "debug             : " ${debug}
    echo "single precision  : " ${single}
    echo "clean build       : " ${cleanup}
    echo "# build threads   : " ${nthreads}
    echo "local mode        : " ${localmode}
    if [ -n "${installpath}" ] ; then
        echo "Install prefix    : " ${installpath}
    else
        echo "Install prefix    : " "(installed locally)"
    fi
    if [ -n "${stelladir}" ] ; then
        echo "STELLA directory  : " ${stelladir}
    else
        echo "STELLA directory  : " "(default)"
    fi
    if [ -n "${nice_cmd}" ] ; then
        echo "Nice              :  ON (really nice)" 
    fi
    echo "============================================================================="
}


cleanupEverything()
{
    # tell user
    echo ">>>>>>>>>>>>>>> cleanup up everything before build"

    # safety check
    if [ ! -f src/dycore/Dycore.h ] ; then
        exitError 701 ${LINENO} "cleanup can only be issued from top level directory"
    fi

    # clean
    /bin/rm -rf build/ install
}


cmakeConfigure()
{
    local install_prefix=$1
    local build_type=$2
    local stella_directory=$3
    local enable_single=$4
    local x87_backend=$5
    local cuda_backend=$6
    local enable_logging=$7
    local enable_gcl=$8
    local enable_meters=$9

    # check validity of arguments
    test -n "${build_type}" || exitError 901 ${LINENO} "Option <build_type> is not set"
    isOnOff "${enable_single}"
    isOnOff "${x86_backend}"
    isOnOff "${cuda_backend}"
    isOnOff "${enable_logging}"
    isOnOff "${enable_gcl}"
    isOnOff "${enable_meters}"

    # check validity of global variables
    test -n "${mpilaunch}" || exitError 921 ${LINENO} "Option <mpilaunch> is not set"
    test -n "${boost_path}" || exitError 922 ${LINENO} "Option <boost_path> is not set"
    test -n "${cuda_arch}" || exitError 923 ${LINENO} "Option <cuda_arch> is not set"
    test -n "${dycore_gpp}" || exitError 924 ${LINENO} "Option <dycore_gpp> is not set"
    test -n "${dycore_gcc}" || exitError 925 ${LINENO} "Option <dycore_gcc> is not set"

    # setup MPI launch configuration (used for unit-tests)
    test -n "${DYCORE_EXECUTE_PREFIX}" || DYCORE_EXECUTE_PREFIX="${mpilaunch}"
    #test -n "${DYCORE_EXECUTE_NUMPROC_FLAG}" || DYCORE_EXECUTE_NUMPROC_FLAG="-n"

    # setup default testdata location
    #test -n "${DYCORE_TESTDATA}" || DYCORE_TESTDATA=${base_path}/build/testdata

    # construct cmake arguments
    local CMAKEARGS=(..
               "-DDYCORE_EXECUTE_PREFIX=${DYCORE_EXECUTE_PREFIX}"
               #"-DDYCORE_TESTDATA=${DYCORE_TESTDATA}"
               "-DCMAKE_C_COMPILER=$(which ${dycore_gcc})"
               "-DCMAKE_CXX_COMPILER=$(which ${dycore_gpp})"
               "-DCMAKE_BUILD_TYPE=${build_type}"
               "-DBoost_INCLUDE_DIR=${boost_path}"
               "-DSINGLEPRECISION=${enable_single}"
               "-DLOGGING=${enable_logging}"
               "-DENABLE_PERFORMANCE_METERS=${enable_meters}"
               "-DGCL=${enable_gcl}"
               "-DENABLE_OPENMP=${dycore_openmp}"
               "-DHORIDIFF_CUDA_COMPUTE_CAPABILITY=${NVIDIA_CUDA_ARCH}"
    )
    if [ ! -z "${stella_directory}" ] ; then
        CMAKEARGS+=("-DSTELLA_DIR=${stella_directory}"
        )
    fi
    if [ ! -z "${install_prefix}" ] ; then
        CMAKEARGS+=("-DCMAKE_INSTALL_PREFIX=${install_prefix}"
        )
    fi
    if [ "${cuda_backend}" == "ON" ] ; then
        CMAKEARGS+=("-DCUDA_BACKEND=ON"
                    "-DDYCORE_CUDA_COMPUTE_CAPABILITY=${cuda_arch}"
        )
    else
        CMAKEARGS+=("-DCUDA_BACKEND=OFF")
    fi

    # setup compilers
    export CXX=${dycore_gpp}
    export CC=${dycore_gcc}

    # launch cmake
    local logfile=`pwd`/cmake.log
    echo ">>>>>>>>>>>>>>> running cmake (see ${logfile})"
    echo cmake "${CMAKEARGS[@]}" 2>&1 1> ${logfile}
    cmake "${CMAKEARGS[@]}" 2>&1 1>> ${logfile}

    # check for Errors and Warnings
    if [ $? -ne 0 ] ; then
        echo "==== START LOG: ${logfile} ===="
        cat ${logfile}
        echo "==== END LOG: ${logfile} ===="
        exitError 820 ${LINENO} "error while running cmake (see log above)"
    fi
    \egrep -i '[^_]error|fail' ${logfile} | egrep -v ' 0 errors|strerror' &>/dev/null
    if [ $? -eq 0 ]; then
        echo "==== START LOG: ${logfile} ===="
        cat ${logfile}
        echo "==== END LOG: ${logfile} ===="
        exitError 821 ${LINENO} "error while running cmake (see log above)"
    fi
    echo ">>>>>>>>>>>>>>>   success"
}


buildLibrary()
{
    # save module environment
    writeModuleList ${base_path}/modules.log loaded "CPP MODULES" ${base_path}/modules_cpp.env

    # check arguments
    local build_dir=${base_path}/build
    if [ ! -d "${build_dir}" ] ; then
        mkdir -p ${build_dir}
    fi

    # check global variables
    test -n "${base_path}" || exitError 801 ${LINENO} "Option <base_path> is not set"
    test -n "${target}" || exitError 802 ${LINENO} "Option <target> is not set"
    test -n "${debug}" || exitError 803 ${LINENO} "Option <debug> is not set"
    test -n "${nthreads}" || exitError 804 ${LINENO} "Option <nthreads> is not set"
    test -n "${dycore_gpp}" || exitError 805 ${LINENO} "Option <dycore_gpp> is not set"
    test -n "${dycore_gcc}" || exitError 806 ${LINENO} "Option <dycore_gcc> is not set"
    test -n "${boost_path}" || exitError 809 ${LINENO} "Option <boost_path> is not set"
    test -n "${cuda_arch}" || exitError 810 ${LINENO} "Option <cuda_arch> is not set"
    test -n "${dycore_openmp}" || exitError 811 ${LINENO} "Option <dycore_openmp> is not set"

    # set build type
    if [ $debug == "ON" ] ; then
        local cmake_build_type=Debug
        enable_logging=ON
    else
        local cmake_build_type=Release
        enable_logging=OFF
    fi
    if [ $single == "ON" ] ; then
        local enable_single=ON
    else
        local enable_single=OFF
    fi
    test -n "${ENABLE_GCL}" || ENABLE_GCL="ON"
    test -n "${ENABLE_PERFORMANCE_METERS}" || ENABLE_PERFORMANCE_METERS="OFF"

    # set backends
    if [ "${target}" == "gpu" ] ; then
      x86_backend=OFF
      cuda_backend=ON 
    else
      x86_backend=ON
      cuda_backend=OFF
    fi

    # show user setup
    echo "--------------------------------------------"
    echo "setup ${library}"
    echo "--------------------------------------------"
    module list
    echo "--------------------------------------------"
    if [ -n "${PE_ENV}" ] ; then echo "Programming Env   :  ${PE_ENV}" ; fi
    echo "C++ compiler      :  `which ${dycore_gpp}`"
    echo "C compiler        :  `which ${dycore_gcc}`"
    echo "Compiler version  :  $(compilerVersion ${dycore_gpp})"
    echo "CMake build type  :  ${cmake_build_type}"
    echo "Single precision  :  ${enable_single}"
    echo "Use CUDA backend  :  ${cuda_backend}"
    if [ "${cuda_backend}" == "ON" ] ; then
        echo "CUDA architecture :  ${cuda_arch}"
    else
        echo "CUDA architecture :  (not applicable)"
    fi
    echo "Use X86 backend   :  ${x86_backend}"
    echo "Boost path        :  ${boost_path}"
    echo "Enable Logging    :  ${enable_logging}"
    echo "Enable GCL        :  ${ENABLE_GCL}"
    echo "Enable Perfmeters :  ${ENABLE_PERFORMANCE_METERS}"
    echo "--------------------------------------------"

    # go to build directory
    cd ${build_dir}

    # configure with cmake if this no cache file exists
    if [ ! -f "CMakeCache.txt" ] ; then
        cmakeConfigure "${installpath}" "${cmake_build_type}" "${stelladir}" "${enable_single}" "${x86_backend}" "${cuda_backend}" "${enable_logging}" "${ENABLE_GCL}" "${ENABLE_PERFORMANCE_METERS}"
    else
        echo "INFORMATION: CMakeCache.txt found... skipping cmakeConfigure"
    fi

    # make and install in this path
    # we have to push stderr to file because the nvcc compiler generates
    # a lot of warnings that can't be turned off
    local logfile=`pwd`/build.log
    echo ">>>>>>>>>>>>>>> building ${library} (see ${logfile})"
    $nice_cmd make -j $nthreads install &> ${logfile}
    if [ $? -eq 0 ] ; then
        echo ">>>>>>>>>>>>>>>   success"
    else
        echo "==== START LOG: ${logfile} ===="
        cat ${logfile}
        echo "==== END LOG: ${logfile} ===="
        exitError 830 ${LINENO} "error while compiling (see log above)"
    fi

    cp "${base_path}/modules_cpp.env" "${installpath}/"
    # return to top directory
    cd ${base_path}

}

##################################################
# setup
##################################################

# echo basic setup
echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"

# start timer
T="$(date +%s)"

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

# load module tools
if [ ! -f ${envloc}/env/moduleTools.sh ] ; then
    exitError 1203 ${LINENO} "could not find ${envloc}/env/moduleTools.sh"
fi
. ${envloc}/env/moduleTools.sh

# setup base path
base_path=$(pwd) # get working base directory

# setup default options (from env/env.${host}.sh)
setupDefaults

# parse command line options (pass all of them to function)
parseOptions $*

# test environment (if requested)
testEnvironment

# only write module files and exit (if requested)
if [ "${writeenv}" == "ON" ] ; then writeCppEnvironment; exit 0; fi

# clean build (if requested)
if [ "${cleanup}" == "ON" ] ; then cleanupEverything ; fi

# display configuration
showBuildConfiguration
writeModuleList ${base_path}/modules.log all "AVAILABLE MODULES"

##################################################
# build Library
##################################################

setCppEnvironment
writeModuleList ${base_path}/modules.log loaded "CPP MODULES" ${base_path}/modules_cpp.env
buildLibrary
unsetCppEnvironment

##################################################
# finish
##################################################

echo "BUILDSUCCESS"
cd $base_path

# end timer and report time taken
T="$(($(date +%s)-T))"
printf "####### time taken: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"

# no errors encountered
echo "####### finished: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
exit 0


