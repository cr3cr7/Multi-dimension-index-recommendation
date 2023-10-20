# check if the right number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 FILE USER IP TARGET"
    exit 1
fi

FILE=$1
# USER=$2
# IP=$3
# TARGET=$4

# Execute the SCP command
scp -P 12366 $FILE chenx@121.48.165.35:/data1/chenx/docker/docker-hadoop-spark/share

