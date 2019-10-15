--------------------------------------------------------------------------------------
-- MATLAB/OCTAVE interface of BudgetedSVM, a Toolbox for Large-scale Non-linear SVM --
--------------------------------------------------------------------------------------

Table of Contents
=================
- Table of Contents
- Introduction
- Installation
- BudgetSVM Usage - Training
- BudgetSVM Usage - Testing
- Other Utilities
- Examples
- Returned Model Structure
- Additional Information


Introduction
============
This tool provides a simple interface to BudgetSVM, a library for large-scale
non-linear, multi-class problems. It is very easy to use as the usage and 
the way of specifying parameters are the same as that of LIBSVM or LIBLINEAR.

Please read the "../GPL.txt" license file before using the BudgetSVM toolbox.


Installation
============
We provide binary files only for 32-bit MATLAB on Windows. If you would 
like to re-build the package, please rely on the following steps.

We recommend using make.m on both MATLAB and OCTAVE. Simply type 'make'
to build 'libsvmread.mex', 'libsvmwrite.mex', 'budgetsvm_train.mex', and
'budgetsvm_predict.mex'.

On MATLAB or Octave type:
	>> make

If make.m does not work on MATLAB (especially for Windows), try 'mex
-setup' to choose a suitable compiler for mex. Make sure your compiler
is accessible and workable. Then type 'make' to start the installation.

Example from author's computer:

	matlab>> mex -setup
	(ps: MATLAB will show the following messages to setup default compiler.)
	Please choose your compiler for building external interface (MEX) files:
	Would you like mex to locate installed compilers [y]/n? y
	Select a compiler:
	[1] Microsoft Visual C/C++ version 7.1 in C:\Program Files\Microsoft Visual Studio
	[0] None
	Compiler: 1
	Please verify your choices:
	Compiler: Microsoft Visual C/C++ 7.1
	Location: C:\Program Files\Microsoft Visual Studio
	Are these correct?([y]/n): y

	matlab>> make

For a list of supported/compatible compilers for MATLAB, please check
the following page:

http://www.mathworks.com/support/compilers/current_release/


BudgetedSVM Usage - Training
============================
In order to train the classification model, run in the Matlab prompt:

>> model = budgetedsvm_train(label_vector, instance_matrix, parameter_string = '');

Inputs:
	label_vector		- label vector of size (N x 1), a label set can include any integer
							representing a class, such as 0/1 or +1/-1 in the case of binary-class
							problems; in the case of multi-class problems it can be any set of integers
	instance_matrix		- instance matrix of size (N x DIMENSIONALITY),
							where each row represents one example
	parameter_string	- parameters of the model, if not provided default empty string is assumed

Output:
	model				- structure that holds the trained model
	

Since the previous call to budgetedsvm_train() function requires the data set to be loaded to Matlab,
which can be infeasible for large data, we provide another variant of the call to the training procedure:

>> budgetedsvm_train(train_file, model_file, parameter_string = '')

Inputs:
	train_file			- filename of .txt file containing training data set in LIBSVM format
	model_file			- filename of .txt file that will contain trained model
	parameter_string	- parameters of the model, defaults to empty string if not provided
	

Parameter string is of the same format for both versions, specified as follows:

	'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'

	The following options are available (default values in parentheses):
	A - algorithm, which large-scale SVM to use (2):
	    0 - Pegasos
	    1 - AMM batch
	    2 - AMM online
	    3 - LLSVM
	    4 - BSGD
	d - dimensionality of the data (required when inputs are .txt files, in that case MUST be set by a user)
	e - number of training epochs in AMM and BSGD (5)
	s - number of subepochs in AMM batch (1)
	b - bias term in AMM, if 0 no bias added (1.0)
	k - pruning frequency, after how many observed examples is pruning done in AMM (10000)
	C - pruning aggresiveness, sets the pruning threshold in AMM, OR
			linear-SVM regularization paramater C in LLSVM (10.0)
	l - limit on the number of weights per class in AMM (20)
	L - learning parameter lambda in AMM and BSGD (0.00010)
	G - kernel width exp(-0.5 * gamma * ||x_i - x_j||^2) in BSGD and LLSVM (1 / DIMENSIONALITY)
	B - total SV set budget in BSGD, OR number of landmard points in LLSVM (100)
	M - budget maintenance strategy in BSGD (0 - removal; 1 - merging), OR
			landmark sampling strategy in LLSVM (0 - random; 1 - k-means; 2 - k-medoids) (1)
	
	z - training and test file are loaded in chunks so that the algorithm can 
			handle budget files on weaker computers; z specifies number of examples loaded in
			a single chunk of data, ONLY when inputs are .txt files (50000)
	w - model weights are split in chunks, so that the algorithm can handle
		highly dimensional data on weaker computers; w specifies number of dimensions stored
		in one chunk, ONLY when inputs are .txt files (1000)
	S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse, used to
			speed up kernel computations (default is 1 when percentage of non-zero
		    features is less than 5%, and 0 when percentage is larger than 5%)
	v - verbose output: 1 to show the algorithm steps (epoch ended, training started, ...), 0 for quiet mode (0)	
	--------------------------------------------

	
BudgetedSVM Usage - Testing
===========================
In order to evaluate the learned model, run in the Matlab prompt the following command:

>> [error_rate, pred_labels] = budgetedsvm_predict(labelVector, instanceMatrix, model, parameter_string);

Inputs:
	labelVector			- label vector of the data set of size (N x 1), a label can be any number							
							representing a class, such as 0/1, or +1/-1, or, in the
							case of multi-class problems, any set of integers
	instanceMatrix		- instance matrix of size (N x DIMENSIONALITY),
							where each row represents one example
	model				- structure holding the model trained using budgetedsvm_train()
	parameter_string	- parameters of the model, if not provided default empty string is assumed

Output:
	error_rate			- error rate on the test set
	pred_labels			- vector of predicted labels of size (N x 1)


Since the previous call to budgetedsvm_predict() function requires the data set to be loaded to Matlab,
we also provide another variant of the call to the testing procedure:

>> [error_rate, pred_labels] = budgetedsvm_predict(test_file, model_file, parameter_string = '')

	Inputs:
		test_file			- filename of .txt file containing test data set in LIBSVM format
		model_file			- filename of .txt file containing model trained through budgetedsvm_train()
		parameter_string	- parameters of the model, defaults to empty string if not provided

	Output:
		error_rate			- error rate on the test set
		pred_labels			- vector of predicted labels of size (N x 1)


Parameter string is of the same format for both versions, specified as follows:

	'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'

	The following options are available (default values in parentheses):
	z - the training and test file are loaded in chunks so that the algorithm can 
			handle budget files on weaker computers; z specifies number of examples loaded in
			a single chunk of data, ONLY when inputs are .txt files (50000)
	w - the model weight is split in parts, so that the algorithm can handle
			highly dimensional data on weaker computers; w specifies number of dimensions stored
			in one chunk, ONLY when inputs are .txt files (1000)
	S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse, used to
			speed up kernel computations (default is 1 when percentage of non-zero
		    features is less than 5%, and 0 when percentage is larger than 5%)
	v - verbose output: 1 to show algorithm steps, 0 for quiet mode (0)
	--------------------------------------------

The function budgetedsvm_predict has two outputs. The first output,
accuracy, is a classification accuracy on the provided testing set.
The second output, predicted_label, is a vector of predicted labels.


Other Utilities
===============
Matlab function libsvmread reads files in LIBSVM format: 

>> [label_vector, instance_matrix] = libsvmread('data.txt'); 

Two outputs are labels and instances, which can then be used as inputs
of budgetsvm_train or budgetsvm_predict functions.

Matlab function libsvmwrite writes Matlab matrix to a file in LIBSVM format:

>> libsvmwrite('data.txt', label_vector, instance_matrix);

The instance_matrix must be a sparse matrix (type must be double).
For 32-bit MATLAB on Windows, pre-built binary files are ready in the directory `../matlab'.

These codes were prepared by Rong-En Fan and Kai-Wei Chang from National Taiwan University.


Examples
========
Here we show a simple example on how to train and test a classifier on the provided adult9a data set.
We first give an example where inputs to budgetedsvm_train and budgetedsvm_predict are data sets first
loaded into Matlab memory, and then provided to BudgetedSVM as Matlab variables:

>> % load the data into Matlab
>> [a9a_label, a9a_inst] = libsvmread('../a9a_train.txt');
>> % train a non-linear model on the training set
>> model = budgetedsvm_train(a9a_label, a9a_inst, '-A 2 -L 0.001 -v 1 -e 5');
>> % evaluate the trained model on the training data
>> [accuracy, predict_label] = budgetedsvm_predict(a9a_label, a9a_inst, model, '-v 1');

Next, we give an example when the inputs to budgetedsvm_train and budgetedsvm_predict are specified
as filenames of files containing training and test data sets, and the model is saved to .txt file:

>> % train a non-linear model on the training set
>> budgetedsvm_train('../a9a_train.txt', '../a9a_model.txt', '-A 2 -L 0.001 -v 1 -e 5 -d 123');
>> % evaluate the trained model on the testing data
>> [accuracy, predict_label] = budgetedsvm_predict('../a9a_test.txt', '../a9a_model.txt', '-v 1');

After running the examples in Matlab prompt, algorithm should return accuracy of around 15%.


Returned Model Structure
========================
The budgetedsvm_train function returns a model which can be used for future
classification. It is a structure organized as follows ["algorithm", "dimension",
"numClasses", " labels", " numWeights", " paramBias", " kernelWidth", "model"]:

	- algorithm		: algorithm used to train a classification model
	- dimension		: dimensionality of the data set
	- numClasses	: number of classes in the data set
	- labels		: label of each class
	- numWeights	: number of weights for each class
	- paramBias		: bias term
	- kernelWidth	: width of the Gaussian kernel
	- model			: the learned model

In order to compress memory and to use the memory efficiently, we coded the model in the following way:

AMM online, AMM batch, and Pegasos: The model is stored as (("dimension" + 1) x "numWeights") matrix. The 
first element of each weight is the degradation of the weight, followed by values of the weight for each 
feature of the data set. If bias term is non-zero, then the final element of each weight corresponds to bias
term, and the matrix is of size (("dimension" + 2) x "numWeights"). By looking at "labels" and "numWeights"
members of Matlab structure we can find out which weights belong to which class. For example, first numWeights[0]
weights belong to labels[0] class, next numWeights[1] weights belong to labels[1] class, and so on.

BSGD: The model is stored as (("numClasses" + "dimension") x "numWeights") matrix. The first "numClasses" 
elements of each weight correspond to alpha parameters for each class, given in order of "labels" member of
the Matlab structure. This is followed by elements of the weights (or support vectors) for each feature of 
the data set.

LLSVM: The model is stored as ((1 + "dimension") x "numWeights") matrix. Each row corresponds to one landmark
point. The first element of each row corresponds to element of linear SVM hyperplane for that particular 
landmark point. This is followed by features of the landmark point in the original feature space. 

More details about the implementation can be found in BudgetedSVM implementation manual
"doc/BudgetedSVM reference manual.pdf" or by openning "doc/html/index.html" in your browser.


Additional Information
======================
The toolbox was written by Nemanja Djuric, Liang Lan, and Slobodan Vucetic
from the Department of Computer and Information Sciences, Temple University,
together with Zhuang Wang from Siemens Corporate Research & Technology.

For any questions, please contact Nemanja Djuric at <nemanja.djuric@temple.edu>.

Acknowledgments:
This work was supported in part by the National Science 
Foundation via grant NSF-IIS-0546155.