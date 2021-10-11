# for i in `seq 0 0.005 0.030`;  
# do
# 	for j in `seq 0 0.005 0.030`;
# 	do
# 		./a.out 0.25 $i $j 0.1 0.1 &
# 	done
# done
for j in `seq 0.000 0.005 0.100`;
do
	./a.out 0.25 $j 0.1 0.1 &
done