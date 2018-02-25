data=../../Intermediate/BBN/
output=../../Results/BBN/
dim=5
lr=0.01
max_iter=2
thread=1
delimter='/'
max_dep=3
word_embed=./shortlist
threshold=-1
for i in `seq 5 5 100`
do
	echo "top k=$i"
	python ./pmi_label.py $data $data $word_embed $i
	#	python ./cwsabie_label.py $data $output
	bash ./warp_label_demo.sh -1000
done
