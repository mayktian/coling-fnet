# run the two demo.sh under src for getting the results with prototype and hierarichal label embeddings


# uncompress data.tar.gz to ./Data to get the raw data for BBN
# uncopmress intermediate.tar.gz to ./Intermediate to get the features extacted for BBN.
# feaures used in this project is reused from the repository of https://github.com/shanzhenren/PLE. They recently released a new version called AFET, but I haven't tested the compatibility with our code.
#example:
bash warp_zero_demo.sh 0.3 3
#threshold=0.3 max_depth=3 during inference.
