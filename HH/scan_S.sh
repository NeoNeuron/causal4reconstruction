# for i in `seq 0 0.005 0.030`;  
# do
# 	for j in `seq 0 0.005 0.030`;
# 	do
# 		./a.out 0.25 $i $j 0.1 0.1 &
# 	done
# done
for j in `seq 0.010 0.001 0.030`;
do
	./a.out 0.25 $j $j 0.1 0.1  >> tmp.log &
done