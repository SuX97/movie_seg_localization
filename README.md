# Problem Definition
Given a video summarization $ v_{s} $, first using HOG video segmentation to divide into shots $ s_{i} $.
Each shots then generates a stack of features and their avg $ 1/N * \sum f_{i} $. The segment is then reformatted to a matrix mxK. $m_{seg}$
In the same way, a long movie is transformed into a feature matrix of MxK. $m_{mov}$

Now we compute the cosine similarity $\frac{m_{seg} \dot m_{mov}}{|m_{seg}|_{2} * |m_{mov}|_{2}} $.

We get the similarity matrix m x M. Using a sliding window to find the sub-sequence with the largest sum of similarity.

Done. The start and point is the left and right boundary of the corder.

# Repo Arch
1. build_feature_database.py
In this file, we save all query segments' feature and original movie/serials feature into lmdb file.

2. shot_segment.py
In this file, we write common shot seperating algorithm such as HOG. Separate a given video into several shorts videos.

3. movie_seg_localization.py
The main file, we reads from the lmdb dataset and calculate the similarity and perform localization.
