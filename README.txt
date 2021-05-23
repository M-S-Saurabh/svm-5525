Name: Saurabh Mylavaram
id: 5593072
email: mylav008@umn.edu

HOW TO RUN THE CODE:
--------------------
Linear SVM on hw2 dataset: 
    python dual_SVM.py

SVM with RBF kernel on hw2 dataset:
    python kernel_SVM.py

Multi-class SVM with both Linear and RBF kernels:
    python multi_SVM.py

In all these files, the __main__ driver function is only three lines, it is easy to see what is being run
Here you can specify the range of C, Sigma values and number of CV splits before running.

ASSUMPTIONS AND NOTES:
---------------------------
- The file 'hw_data_2020.csv' is present in the same directory as these files.

- The extracted 'mfeat' directory is also present in the same directory.

- Since 'SVM with Linear Kernel' in problem 3 is the same as SVM with no kernel in problem 2,
  I only run it once in dual_SVM.py. In kernel_SVM.py it only runs the RBF kernel and shows results.

- Since there are 64 combinations of C, gamma values possible in problem 3, I did not list results of
  all those combinations in the pdf submission, instead a copy of command-line outputs from all files
  are listed in a separate text file named: 'CLI-outputs.txt'.
  They are also printed out on command-line when the files are run.

- All three files directly/indirectly use the same implementation of `kernelSVM` class written in the file 'SVM.py'.


