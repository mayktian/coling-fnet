for i in `seq 1 1 20`
do
		echo "random seed $i"
		python ./test_Wsabie_label.py $i 0
		bash ./warp_py_demo.sh 
		echo "testing Wsabie + TransE"
		python ./test_Wsabie_label.py $i 1
		bash ./warp_py_demo.sh
done
