for i in `seq 0.01 0.01 0.2`
do
		python ./test_label_corr.py $i
		bash ./warp_py_demo.sh 
done
