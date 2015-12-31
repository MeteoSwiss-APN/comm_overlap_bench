#!/bin/bash -f

# This is a harness for testing the jenkins.sh script. It needs to setup several variables
# which are usually automatically set by jenkins before launching the script.

export slave="lema"
export target="cpu"
export build_type="Release"
export real_type="double"
#BUILD_NUMBER       The current build number, such as "153".
#BUILD_ID           The current build id, such as "2005-08-22_23-59-59" (YYYY-MM-DD_hh-mm-ss).
#BUILD_DISPLAY_NAME The display name of the current build, something like "#153" by default.
export JOB_NAME="autobuild_branch_test/slave=lema"
export BUILD_TAG="jenkins-autobuild_branch_test-slave=lema-26"
#EXECUTOR_NUMBER    The unique number that identifies the current executor.
#NODE_NAME          Name of the slave if the build is on a slave, or "master" if run on master.
#NODE_LABELS        Whitespace-separated list of labels that the node is assigned.
export WORKSPACE="/scratch/jenkins/workspace/autobuild_branch_test/slave/lema"
#JENKINS_HOME       The absolute path of the data storage directory assigned on the master node.
#JENKINS_URL        Full URL of Jenkins, like http://server:port/jenkins/
#BUILD_URL          Full URL of this build, like http://server:port/jenkins/job/foo/15/
#JOB_URL            Full URL of this job, like http://server:port/jenkins/job/foo/
#SVN_REVISION       Subversion revision number that's currently checked out to the workspace.
#SVN_URL            Subversion URL that's currently checked out to the workspace.

`dirname $0`/jenkins.sh $*

# goodbye, stranger!
