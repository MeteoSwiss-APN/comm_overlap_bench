# Download the env folder
##############################################################################################

`module load git; git submodule init &>/dev/null && git submodule update --remote &>/dev/null`
if [ $? -ne 0 ] ; then
  echo "WARNING: problem checking out environment files. Defaulting to offline mode."
fi

# Check if the env directory exists
if [ ! -d "${envloc}/env" ] ; then
  # We don't have the machine environment yet.
  echo "Error: The env directory ${envloc}/env does not exist. Aborting."
  exit 1
fi