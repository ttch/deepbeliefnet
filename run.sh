export CURRENT=$(pwd)
sudo nvidia-docker run -it \
-v $CURRENT:/mlnet \
-v $HOME/spark-2.3.0-bin-hadoop2.7:/spark-2.3.0-bin-hadoop2.7 \
-v $HOME/work/deepbeliefnet/data:/data \
dbn-gpu-tensorflow /bin/bash
