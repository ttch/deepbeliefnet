#/spark-2.3.0-bin-hadoop2.7/bin/spark-submit main.py \
#--driver-memory 32g \
#--executor-memory 2g \
#--executor-cores 16
python main.py

let total=10#`cat args.cfg |wc -l`-1

for ((i=0; i<$total; i ++))
do
	/spark-2.3.0-bin-hadoop2.7/bin/spark-submit task.py $i \
	--driver-memory 32g \
	--executor-memory 2g \
	--executor-cores 16
done