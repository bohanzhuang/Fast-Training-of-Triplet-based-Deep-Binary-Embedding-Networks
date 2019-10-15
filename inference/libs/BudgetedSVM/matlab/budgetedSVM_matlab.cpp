/*
	\file budgetedSVM_matlab.cpp
	\brief Implements classes and functions that are used to communicate between C++ and Matlab environment.
*/
/* 
	* This program is a free software; you can redistribute it and/or modify
	* it under the terms of the GNU General Public License as published by
	* the Free Software Foundation; either version 3 of the License, or
	* (at your option) any later version.
	
	Author	:	Nemanja Djuric, with some parts influenced by LIBSVM C++ code
	Name	:	budgetedSVM_matlab.cpp
	Date	:	December 10th, 2012
	Desc.	:	Implements classes and functions that are used to communicate between C++ and Matlab environment.
	Version	:	v1.01
*/

#include "../Eigen/Dense"
using namespace Eigen;

#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>
using namespace std;

#include "mex.h"

#include "../src/budgetedSVM.h"
#include "../src/mm_algs.h"
#include "../src/bsgd.h"
#include "../src/llsvm.h"
#include "budgetedSVM_matlab.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

/*!
    \brief Expected number of fields in Matlab structure from which the model is loaded.
*/
#define NUM_OF_RETURN_FIELD 8

/*!
    \brief Labels of the fields in Matlab structure.
*/
static const char *fieldNames[] = 
{
	"algorithm",
	"dimension",
	"numClasses",
	"labels",
	"numWeights",
	"paramBias",
	"kernelWidth",
	"model",
};
	
/* \fn static int getAlgorithm(const mxArray *matlabStruct)
	\brief Get algorithm from the trained model stored in Matlab structure.
	\param [in] matlabStruct Pointer to Matlab structure.
	\return -1 if error, otherwise returns algorithm code from the model file.
*/
int budgetedModelMatlab::getAlgorithm(const mxArray *matlabStruct)
{	
	if (mxGetNumberOfFields(matlabStruct) != NUM_OF_RETURN_FIELD)
		return -1;
	
	// get algorithm
	return (int)(*(mxGetPr(mxGetFieldByNumber(matlabStruct, 0, 0))));
}

/* \fn void budgetedDataMatlab::readDataFromMatlab(const mxArray *labelVec, const mxArray *instanceMat, parameters *param)
	\brief Loads the data from Matlab.
	\param [in] labelVec Vector of labels.
	\param [in] instanceMat Matrix of data points, each row is a single data point.
	\param [in] param The parameters of the algorithm.
*/	
void budgetedDataMatlab::readDataFromMatlab(const mxArray *labelVec, const mxArray *instanceMat, parameters *param)
{
	long start = clock();
	unsigned int i, j, k, labelVectorRowNum;
	long unsigned int low, high;
	mwIndex *ir, *jc;
	double *samples, *labels;
	bool labelFound;
	mxArray *instanceMatCol; // transposed instance sparse matrix
		
	// otherwise load the data, given below	
	// transpose instance matrix
	{
		mxArray *prhs[1], *plhs[1];
		prhs[0] = mxDuplicateArray(instanceMat);
		if (mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
			mexErrMsgTxt("Error: Cannot transpose training instance matrix.\n");
		
		instanceMatCol = plhs[0];
		mxDestroyArray(prhs[0]);
	}

	// each column is one instance
	labels = mxGetPr(labelVec);
	samples = mxGetPr(instanceMatCol);

	// get number of instances
	labelVectorRowNum = (int)mxGetM(labelVec);
	if (labelVectorRowNum != (int)mxGetN(instanceMatCol))
		mexErrMsgTxt("Length of label vector does not match number of instances.\n");

	// set the dimension and the number of data points
	this->N = labelVectorRowNum;
	if ((*param).DIMENSION == 0)
	{
		// it is 0 when loading training data set
		this->dimension = (*param).DIMENSION = (int)mxGetM(instanceMatCol);
		if ((*param).BIAS_TERM != 0.0)
			(*param).DIMENSION++;
		
		// set GAMMA_PARAM here if needed, done during loading of training set
		if ((*param).GAMMA_PARAM == 0.0)
			(*param).GAMMA_PARAM = 1.0 / (double) (*param).DIMENSION;
	}
	else
	{
		// it is non-zero only when loading testing data set, no need to set GAMMA parameter as it is read from the model structure from Matlab
		this->dimension = (*param).DIMENSION;	
		
		// if bias term is non-zero, then the actual dimensionality of data is one less than DIMENSION
		if ((*param).BIAS_TERM != 0.0)
			this->dimension--;					
	}

	// allocate memory for labels
	this->al = new (nothrow) unsigned char[this->N];
	if (this->al == NULL)
		mexErrMsgTxt("Memory allocation error (readDataFromMatlab function)! Restart MATLAB and try again.");
	
	if (mxIsSparse(instanceMat))
	{
		ir = mxGetIr(instanceMatCol);
		jc = mxGetJc(instanceMatCol);
		
		j = 0;				
		for (i = 0; i < labelVectorRowNum; i++)
		{
			// where the instance starts
			ai.push_back(j);
			
			// get yLabels, if label not seen before add it in the label array
			labelFound = false;
			for (k = 0; k < (int) yLabels.size(); k++)
			{
				if (yLabels[k] == (int)labels[i])
				{
					al[i] = k;
					labelFound = true;
					break;
				}
			}
			if (!labelFound)
			{						
				yLabels.push_back((int)labels[i]);
				al[i] = yLabels.size() - 1;
			}
			
			// get features
			low = (int)jc[i], high = (int)jc[i + 1];
			for (k = low; k < high; k++)
			{
				// we save the actual feature no. in aj, and the value in an
				aj.push_back((int)ir[k] + 1);
				an.push_back((float) samples[k]);
				j++;					
			}
		}
	}
	else
	{
		j = 0;
		low = 0;
		for (i = 0; i < labelVectorRowNum; i++)
		{
			// where the instance starts
			ai.push_back(j);
			
			// get yLabels, if label not seen before add it in the label array
			labelFound = false;
			for (k = 0; k < (int) yLabels.size(); k++)
			{
				if (yLabels[k] == (int)labels[i])
				{
					al[i] = k;
					labelFound = true;
					break;
				}
			}
			if (!labelFound)
			{						
				yLabels.push_back((int)labels[i]);
				al[i] = yLabels.size() - 1;
			}
			
			// get features
			for (k = 0; k < (int)mxGetM(instanceMatCol); k++)
			{
				if (samples[low] != 0.0)
				{
					// we save the actual feature no. in aj, and the value in an
					aj.push_back(k + 1);
					an.push_back((float) samples[low]);
					j++;
				}
				low++;
			}
		}
	}
	
	// if very beginning, just allocate memory for assignments
	if (keepAssignments)
		this->assignments = new (nothrow) unsigned int[this->N];
	
	loadTime += (clock() - start);
};

/* \fn void budgetedModelMatlab::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
	\brief Save the trained model to Matlab, by creating Matlab structure.
	\param [out] plhs Pointer to Matlab output.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
*/
void budgetedModelMatlabAMM::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
{
	unsigned int i, j, numWeights = 0, cnt;
	double *ptr;
	mxArray *returnModel, **rhs;
	int outID = 0;
	
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * NUM_OF_RETURN_FIELD);

	// algorithm type
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->ALGORITHM;
	outID++;
	
	// dimension
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->DIMENSION;
	outID++;
	
	// number of classes
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = (*yLabels).size();
	outID++;
	
	// labels
	rhs[outID] = mxCreateDoubleMatrix((*yLabels).size(), 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	for (i = 0; i < (*yLabels).size(); i++)
		ptr[i] = (*yLabels)[i];
	outID++;
	
	// total number of weights
	rhs[outID] = mxCreateDoubleMatrix((*yLabels).size(), 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	for (i = 0; i < (*modelMM).size(); i++)
	{
		ptr[i] = (*modelMM)[i].size();
		numWeights += (*modelMM)[i].size();
	}
	outID++;
	
	// bias param
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->BIAS_TERM;
	outID++;
	
	// kernel width
	rhs[outID] = mxCreateDoubleMatrix(0, 0, mxREAL);
	outID++;
	
	// weights
	int irIndex, nonZeroElement;
	mwIndex *ir, *jc;
	
	// find how many non-zero elements there are
	nonZeroElement = 0;
	for (i = 0; i < (*modelMM).size(); i++) 
	{
		for (j = 0; j < (*modelMM)[i].size(); j++)
		{
			for (unsigned int k = 0; k < (*param).DIMENSION; k++)              // for every feature
			{
				if ((*((*modelMM)[i][j]))[k] != 0.0)
					nonZeroElement++;
			}
		}
	}
	
	// +1 is for degradation of AMM algorithms, it will be the first number in the row representing a weight
	if (param->ALGORITHM == PEGASOS)
		rhs[outID] = mxCreateSparse(param->DIMENSION, numWeights, nonZeroElement, mxREAL);
	else if ((param->ALGORITHM == AMM_BATCH) || (param->ALGORITHM == AMM_ONLINE))
		rhs[outID] = mxCreateSparse(param->DIMENSION + 1, numWeights, nonZeroElement + numWeights, mxREAL);
	ir = mxGetIr(rhs[outID]);
	jc = mxGetJc(rhs[outID]);
	ptr = mxGetPr(rhs[outID]);
	jc[0] = irIndex = cnt = 0;		
	for (i = 0; i < (*modelMM).size(); i++)
	{
		for (j = 0; j < (*modelMM)[i].size(); j++)
		{
			int xIndex = 0;
			
			// this adds degradation to the beginning of a vector, more compact 
			if ((param->ALGORITHM == AMM_BATCH) || (param->ALGORITHM == AMM_ONLINE))
			{
				ir[irIndex] = 0; 
				ptr[irIndex] = (*modelMM)[i][j]->getDegradation();
				irIndex++, xIndex++;
			}
			
			// add the actual features
			for (unsigned int k = 0; k < (*param).DIMENSION; k++)              // for every feature
			{
				if ((*((*modelMM)[i][j]))[k] != 0.0)
				{
					if ((param->ALGORITHM == AMM_BATCH) || (param->ALGORITHM == AMM_ONLINE))
						ir[irIndex] = k + 1;
					else if (param->ALGORITHM == PEGASOS)
						ir[irIndex] = k;
					ptr[irIndex] = (*((*modelMM)[i][j]))[k];
					irIndex++, xIndex++;
				}
			}
			jc[cnt + 1] = jc[cnt] + xIndex;
			cnt++;
		}			
	}
	// commented, since now it is appended to the weight matrix
	/*// degradations
	cnt = 0;
	rhs[outID] = mxCreateDoubleMatrix(numWeights, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	for (i = 0; i < (*modelMM).size(); i++)
		for (j = 0; j < (*modelMM)[i].size(); j++)
			ptr[cnt++] = (*modelMM)[i][j]->degradation;
	outID++;*/
		
	/* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
	returnModel = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, fieldNames);
	
	/* Fill struct matrix with input arguments */
	for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
		mxSetField(returnModel, 0, fieldNames[i], mxDuplicateArray(rhs[i]));
	
	plhs[0] = returnModel;
	mxFree(rhs);
}

/* \fn bool budgetedModelMatlabAMM::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg);
	\brief Loads the trained model from Matlab structure.
	\param [in] matlabStruct Pointer to Matlab structure.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
	\param [out] msg Error message, if error encountered.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelMatlabAMM::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg)
{
	int i, j, numOfFields, numClasses, currClass, classCounter;
	double *ptr;
	int id = 0;
	mxArray **rhs;
	vector <unsigned int> numWeights;
	double sqrNorm;
	
	numOfFields = mxGetNumberOfFields(matlabStruct);
	if (numOfFields != NUM_OF_RETURN_FIELD) 
	{
		*msg = "number of return fields is not correct";
		return false;
	}
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * numOfFields);

	for (i = 0; i < numOfFields; i++)
		rhs[i] = mxGetFieldByNumber(matlabStruct, 0, i);	
	
	// algorithm
	ptr = mxGetPr(rhs[id]);
	param->ALGORITHM = (unsigned int)ptr[0];
	id++;
	
	// dimension
	ptr = mxGetPr(rhs[id]);
	param->DIMENSION = (unsigned int)ptr[0];
	id++;
	
	// numClasses
	ptr = mxGetPr(rhs[id]);
	numClasses = (unsigned int)ptr[0];
	id++;
	
	// labels
	if (mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		for(i = 0; i < numClasses; i++)
		{
			(*yLabels).push_back((int)ptr[i]);
			
			// add to model empty weight vector for each class
			vector <budgetedVectorAMM*> tempV;
			(*modelMM).push_back(tempV);
		}
	}
	id++;
	
	// numWeights
	if (mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		for(i = 0; i < numClasses; i++)
		{
			numWeights.push_back((int)ptr[i]);
		}
	}
	id++;
	
	// bias term
	ptr = mxGetPr(rhs[id]);
	param->BIAS_TERM = (double)ptr[0];
	id++;
	
	// kernel width next, just skip
	id++;
	
	// weights
	int sr, sc;
	mwIndex *ir, *jc;

	sr = (int)mxGetN(rhs[id]);
	sc = (int)mxGetM(rhs[id]);

	ptr = mxGetPr(rhs[id]);
	ir = mxGetIr(rhs[id]);
	jc = mxGetJc(rhs[id]);
	
	// weights are in columns
	currClass = classCounter = 0;
	for (i = 0; i < sr; i++)
	{
		int low = (int)jc[i], high = (int)jc[i + 1];			
		budgetedVectorAMM *eNew = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
		sqrNorm = 0.0;
		
		for (j = low; j < high; j++)
		{
			if (param->ALGORITHM == PEGASOS)
				((*eNew)[(int)ir[j]]) = (float)ptr[j];
			else if ((param->ALGORITHM == AMM_BATCH) || (param->ALGORITHM == AMM_ONLINE))
			{
				if (j == low)
					eNew->setDegradation(ptr[j]);
				else
				{
					((*eNew)[(int)ir[j] - 1]) = (float)ptr[j];
					sqrNorm += (ptr[j] * ptr[j]);
				}
			}
		}
		eNew->setSqrL2norm(sqrNorm);
		(*modelMM)[currClass].push_back(eNew);			
		eNew = NULL;
		
		// increment weight counter and check if new class is starting
		if (++classCounter == numWeights[currClass])
		{
			classCounter = 0;
			currClass++;		
		}
	}
	id++;
	
	mxFree(rhs);
	return true;
}

/* \fn void budgetedModelMatlabBSGD::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
	\brief Save the trained model to Matlab, by creating Matlab structure.
	\param [out] plhs Pointer to Matlab output.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
*/
void budgetedModelMatlabBSGD::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
{
	unsigned int i, j, numWeights = 0, cnt;
	double *ptr;
	mxArray *returnModel, **rhs;
	int outID = 0;
	
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * NUM_OF_RETURN_FIELD);

	// algorithm type
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->ALGORITHM;
	outID++;
	
	// dimension
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->DIMENSION;
	outID++;
	
	// number of classes
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = (*yLabels).size();
	outID++;
	
	// labels
	rhs[outID] = mxCreateDoubleMatrix((*yLabels).size(), 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	for (i = 0; i < (*yLabels).size(); i++)
		ptr[i] = (*yLabels)[i];
	outID++;
	
	// total number of weights
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = (*modelBSGD).size();
	numWeights = (*modelBSGD).size();
	outID++;
	
	// bias param
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->BIAS_TERM;
	outID++;
	
	// kernel width
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->GAMMA_PARAM;
	outID++;
	
	// weights, different for MM algorithms, BSGD and LLSVM
	int irIndex, nonZeroElement;
	mwIndex *ir, *jc;
	
	// find how many non-zero elements there are
	nonZeroElement = 0;
	for (i = 0; i < (*modelBSGD).size(); i++) 
	{
		// count non-zero features
		for (j = 0; j < (*param).DIMENSION; j++)
		{
			if ((*((*modelBSGD)[i]))[j] != 0.0)
				nonZeroElement++;
		}
		
		// count non-zero alphas also
		for (j = 0; j < (*yLabels).size(); j++)
		{
			if ((*((*modelBSGD)[i])).alphas[j] != 0.0)
				nonZeroElement++;
		}
	}

	//  +(*yLabels).size() is for the alpha parameters of each BSGD weight
	rhs[outID] = mxCreateSparse(param->DIMENSION + (*yLabels).size(), numWeights, nonZeroElement, mxREAL);
	ir = mxGetIr(rhs[outID]);
	jc = mxGetJc(rhs[outID]);
	ptr = mxGetPr(rhs[outID]);
	jc[0] = irIndex = cnt = 0;	
	for (i = 0; i < (*modelBSGD).size(); i++)
	{
		int xIndex = 0;
		
		// this adds alpha weights to the beginning of a vector, more compact
		for (j = 0; j < (*yLabels).size(); j++)
		{
			if ((*((*modelBSGD)[i])).alphas[j] != 0.0)
			{
				ir[irIndex] = j; 
				ptr[irIndex] = (*((*modelBSGD)[i])).alphas[j];
				irIndex++, xIndex++;
			}
		}
		
		// add the actual features
		for (j = 0; j < (*param).DIMENSION; j++)              // for every feature
		{
			if ((*((*modelBSGD)[i]))[j] != 0.0)
			{
				ir[irIndex] = j + (*yLabels).size();		// shift it to accomodate alpha weights
				ptr[irIndex] = (*((*modelBSGD)[i]))[j];
				irIndex++, xIndex++;
			}
		}
		jc[cnt + 1] = jc[cnt] + xIndex;
		cnt++;
	}
	
	/* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
	returnModel = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, fieldNames);
	
	/* Fill struct matrix with input arguments */
	for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
		mxSetField(returnModel, 0, fieldNames[i], mxDuplicateArray(rhs[i]));
	
	plhs[0] = returnModel;
	mxFree(rhs);
}

/* \fn bool budgetedModelMatlabBSGD::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg);
	\brief Loads the trained model from Matlab structure.
	\param [in] matlabStruct Pointer to Matlab structure.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
	\param [out] msg Error message, if error encountered.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelMatlabBSGD::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg)
{
	int i, j, numOfFields, numClasses, currClass, classCounter;
	double *ptr;
	int id = 0;
	mxArray **rhs;
	vector <unsigned int> numWeights;
	double sqrNorm;
	
	numOfFields = mxGetNumberOfFields(matlabStruct);
	if (numOfFields != NUM_OF_RETURN_FIELD) 
	{
		*msg = "number of return fields is not correct";
		return false;
	}
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * numOfFields);

	for (i = 0; i < numOfFields; i++)
		rhs[i] = mxGetFieldByNumber(matlabStruct, 0, i);	
	
	// algorithm
	ptr = mxGetPr(rhs[id]);
	param->ALGORITHM = (unsigned int)ptr[0];
	id++;
	
	// dimension
	ptr = mxGetPr(rhs[id]);
	param->DIMENSION = (unsigned int)ptr[0];
	id++;
	
	// numClasses
	ptr = mxGetPr(rhs[id]);
	numClasses = (unsigned int)ptr[0];
	id++;
	
	// labels
	if (mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		for(i = 0; i < numClasses; i++)
			(*yLabels).push_back((int)ptr[i]);
	}
	id++;
	
	// numWeights, just skip
	id++;
	
	// bias term
	ptr = mxGetPr(rhs[id]);
	param->BIAS_TERM = (double)ptr[0];
	id++;
	
	// kernel width
	ptr = mxGetPr(rhs[id]);
	param->GAMMA_PARAM = (double)ptr[0];
	id++;
	
	// weights
	int sr, sc;
	mwIndex *ir, *jc;

	sr = (int)mxGetN(rhs[id]);
	sc = (int)mxGetM(rhs[id]);

	ptr = mxGetPr(rhs[id]);
	ir = mxGetIr(rhs[id]);
	jc = mxGetJc(rhs[id]);
	
	// weights are in columns
	currClass = classCounter = 0;
	for (i = 0; i < sr; i++)
	{
		int low = (int)jc[i], high = (int)jc[i + 1];			
		budgetedVectorBSGD *eNew = new budgetedVectorBSGD((*param).DIMENSION, (*param).CHUNK_WEIGHT, numClasses);
		sqrNorm = 0.0;
		
		for (j = low; j < high; j++)
		{
			if ((unsigned int)ir[j] < (*yLabels).size())
			{
				// get alpha values
				eNew->alphas[(int)ir[j]] = ptr[j];
			}
			else
			{
				// get features
				((*eNew)[(int)ir[j] - (*yLabels).size()]) = (float)ptr[j];
				sqrNorm += (ptr[j] * ptr[j]);
			}
		}
		eNew->setSqrL2norm(sqrNorm);
		(*modelBSGD).push_back(eNew);			
		eNew = NULL;
	}
	id++;
	
	mxFree(rhs);
	return true;
}

/* \fn void budgetedModelMatlabLLSVM::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
	\brief Save the trained model to Matlab, by creating Matlab structure.
	\param [out] plhs Pointer to Matlab output.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
*/
void budgetedModelMatlabLLSVM::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
{
	unsigned int i, j, numWeights = 0, cnt;
	double *ptr;
	mxArray *returnModel, **rhs;
	int outID = 0;
	
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * NUM_OF_RETURN_FIELD);

	// algorithm type
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->ALGORITHM;
	outID++;
	
	// dimension
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->DIMENSION;
	outID++;
	
	// number of classes
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = (*yLabels).size();
	outID++;
	
	// labels
	rhs[outID] = mxCreateDoubleMatrix((*yLabels).size(), 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	for (i = 0; i < (*yLabels).size(); i++)
		ptr[i] = (*yLabels)[i];
	outID++;
	
	// total number of weights
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = (*modelLLSVMlandmarks).size();
	numWeights = (*modelLLSVMlandmarks).size();
	outID++;
	
	// bias param
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->BIAS_TERM;
	outID++;
	
	// kernel width
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->GAMMA_PARAM;
	outID++;
	
	// weights, different for MM algorithms, BSGD and LLSVM
	int irIndex, nonZeroElement;
	mwIndex *ir, *jc;
	
	// find how many non-zero elements there are
	nonZeroElement = 0;
	for (i = 0; i < (*modelLLSVMlandmarks).size(); i++) 
	{
		// count non-zero features
		for (j = 0; j < (*param).DIMENSION; j++)
		{
			if ((*((*modelLLSVMlandmarks)[i]))[j] != 0.0)
				nonZeroElement++;
		}
		
		// count all elements of modelLLSVMmatrixW also
		nonZeroElement += (numWeights * numWeights);
		
		// count linear SVM length also
		nonZeroElement += numWeights;
	}

	//  +(*yLabels).size() is for the alpha parameters of each BSGD weight
	rhs[outID] = mxCreateSparse(param->DIMENSION + numWeights + 1, numWeights, nonZeroElement, mxREAL);
	ir = mxGetIr(rhs[outID]);
	jc = mxGetJc(rhs[outID]);
	ptr = mxGetPr(rhs[outID]);
	jc[0] = irIndex = cnt = 0;	
	for (i = 0; i < (*modelLLSVMlandmarks).size(); i++)
	{
		int xIndex = 0;
		
		// this adds alpha weights to the beginning of a vector, more compact
		ir[irIndex] = 0; 
		ptr[irIndex] = modelLLSVMweightVector(i, 0);
		irIndex++, xIndex++;
		
		// this adds row of modelLLSVMmatrixW next, more compact
		for (j = 0; j < numWeights; j++)
		{
			ir[irIndex] = j + 1;		// shift it to accomodate linear weight
			ptr[irIndex] = modelLLSVMmatrixW(i, j);
			irIndex++, xIndex++;
		}
		
		// add the actual features
		for (j = 0; j < (*param).DIMENSION; j++)
		{
			if ((*((*modelLLSVMlandmarks)[i]))[j] != 0.0)
			{
				ir[irIndex] = j + numWeights + 1;		// shift it to accomodate linear weight and modelLLSVMmatrixW
				ptr[irIndex] = (*((*modelLLSVMlandmarks)[i]))[j];
				irIndex++, xIndex++;
			}
		}
		jc[cnt + 1] = jc[cnt] + xIndex;
		cnt++;
	}
	
	/* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
	returnModel = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, fieldNames);
	
	/* Fill struct matrix with input arguments */
	for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
		mxSetField(returnModel, 0, fieldNames[i], mxDuplicateArray(rhs[i]));
	
	plhs[0] = returnModel;
	mxFree(rhs);
}

/* \fn bool budgetedModelMatlabLLSVM::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg);
	\brief Loads the trained model from Matlab structure.
	\param [in] matlabStruct Pointer to Matlab structure.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
	\param [out] msg Error message, if error encountered.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelMatlabLLSVM::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg)
{
	unsigned int i, j, numOfFields, numClasses;
	double *ptr, sqrNorm;
	int id = 0;
	mxArray **rhs;
	
	numOfFields = mxGetNumberOfFields(matlabStruct);
	if (numOfFields != NUM_OF_RETURN_FIELD) 
	{
		*msg = "Number of return fields is not correct.";
		return false;
	}
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * numOfFields);

	for (i = 0; i < numOfFields; i++)
		rhs[i] = mxGetFieldByNumber(matlabStruct, 0, i);	
	
	// algorithm
	ptr = mxGetPr(rhs[id]);
	param->ALGORITHM = (unsigned int)ptr[0];
	id++;
	
	// dimension
	ptr = mxGetPr(rhs[id]);
	param->DIMENSION = (unsigned int)ptr[0];
	id++;
	
	// numClasses
	ptr = mxGetPr(rhs[id]);
	numClasses = (unsigned int)ptr[0];
	id++;
	
	// labels
	if (mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		for(i = 0; i < numClasses; i++)
			(*yLabels).push_back((int)ptr[i]);
	}
	id++;
	
	// numWeights
	ptr = mxGetPr(rhs[id]);
	param->BUDGET_SIZE = (unsigned int)ptr[0];
	id++;
	
	// bias term
	ptr = mxGetPr(rhs[id]);
	param->BIAS_TERM = (double)ptr[0];
	id++;
	
	// kernel width
	ptr = mxGetPr(rhs[id]);
	param->GAMMA_PARAM = (double)ptr[0];
	id++;
	
	// weights
	unsigned int sr, sc;
	mwIndex *ir, *jc;

	sr = (int)mxGetN(rhs[id]);
	sc = (int)mxGetM(rhs[id]);

	ptr = mxGetPr(rhs[id]);
	ir = mxGetIr(rhs[id]);
	jc = mxGetJc(rhs[id]);
	
	// allocate memory for model
	modelLLSVMmatrixW.resize((*param).BUDGET_SIZE, (*param).BUDGET_SIZE);
	modelLLSVMweightVector.resize((*param).BUDGET_SIZE, 1);
	
	// weight-vectors are in columns
	for (i = 0; i < sr; i++)
	{
		unsigned int low = (int)jc[i], high = (int)jc[i + 1];			
		budgetedVectorLLSVM *eNew = new budgetedVectorLLSVM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
		sqrNorm = 0.0;
		
		// get the linear weight
		modelLLSVMweightVector(i, 0) = ptr[low];
		
		// get the modelLLSVMmatrixW
		for (j = low + 1; j < low + (*param).BUDGET_SIZE + 1; j++)
			modelLLSVMmatrixW(i, j - low - 1) = ptr[j];
		
		// get the features
		for (j = low + (*param).BUDGET_SIZE + 1; j < high; j++)
		{
			((*eNew)[(int)ir[j] - (*param).BUDGET_SIZE - 1]) = (float)ptr[j];
			sqrNorm += (ptr[j] * ptr[j]);
		}
		eNew->setSqrL2norm(sqrNorm);
		(*modelLLSVMlandmarks).push_back(eNew);			
		eNew = NULL;
	}
	id++;
	
	mxFree(rhs);
	return true;
}

/* \fn void printStringMatlab(const char *s) 
	\brief Prints string to Matlab, used to modify callback found in budgetedSVM.cpp
	\param [in] s Text to be printed.
*/
void printStringMatlab(const char *s) 
{
	mexPrintf(s);
	mexEvalString("drawnow;");
}

/* \fn void printErrorStringMatlab(const char *s) 
	\brief Prints error string to Matlab, used to modify callback found in budgetedSVM.cpp
	\param [in] s Text to be printed.
*/
void printErrorStringMatlab(const char *s) 
{
	mexErrMsgTxt(s);
}

/* \fn void fakeAnswer(mxArray *plhs[])
	\brief Returns empty matrix to Matlab.
	\param [out] plhs Pointer to Matlab output.
*/
void fakeAnswer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

/* \fn void printUsageMatlab(bool trainingPhase)
	\brief Prints to standard output the instructions on how to use the software.
	\param [in] trainingPhase Indicator if training or testing phase.
*/
void printUsageMatlab(bool trainingPhase, parameters *param)
{
	if (trainingPhase)
	{
		mexPrintf("\n\tUsage:\n");
		mexPrintf("\t\tmodel = budgetedsvm_train(label_vector, instance_matrix, parameter_string = '')\n\n");
		mexPrintf("\tInputs:\n");
		mexPrintf("\t\tlabel_vector\t\t- label vector of size (N x 1), a label set can include any integer\n");
		mexPrintf("\t\t\t\t\t              representing a class, such as 0/1 or +1/-1 in the case of binary-class\n");
		mexPrintf("\t\t\t\t\t              problems; in the case of multi-class problems it can be any set of integers\n");
		mexPrintf("\t\tinstance_matrix\t\t- instance matrix of size (N x DIMENSIONALITY),\n");
		mexPrintf("\t\t\t\t                  where each row represents one example\n");
		mexPrintf("\t\tparameter_string\t- parameters of the model, defaults to empty string if not provided\n\n");
		mexPrintf("\tOutput:\n");
		mexPrintf("\t\tmodel\t\t\t\t- structure that holds the learned model\n\n");
		
		mexPrintf("\t--------------------------------------------\n\n");
		mexPrintf("\tIf the data set cannot be fully loaded to Matlab, another variant can be used:\n");
		mexPrintf("\t\tbudgetedsvm_train(train_file, model_file, parameter_string = '')\n\n");
		mexPrintf("\tInputs:\n");
		mexPrintf("\t\ttrain_file\t\t\t- filename of .txt file containing training data set in LIBSVM format\n");
		mexPrintf("\t\tmodel_file\t\t\t- filename of .txt file that will contain trained model\n");
		mexPrintf("\t\tparameter_string\t- parameters of the model, defaults to empty string if not provided\n\n");
		
		mexPrintf("\t--------------------------------------------\n\n");
		mexPrintf("\tParameter string is of the following format:\n");
		mexPrintf("\t'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'\n\n");
		mexPrintf("\tThe following options are available (default values in parentheses):\n");
		mexPrintf("\t A - algorithm, which budget-scale SVM to use (%d):\n", (*param).ALGORITHM);
		mexPrintf("\t\t     0 - Pegasos\n");
		mexPrintf("\t\t     1 - AMM batch\n");
		mexPrintf("\t\t     2 - AMM online\n");
		mexPrintf("\t\t     3 - LLSVM\n");
		mexPrintf("\t\t     4 - BSGD\n");
		mexPrintf("\t d - dimensionality of the data (required when inputs are .txt files, in that case MUST be set by a user)\n");
		mexPrintf("\t e - number of training epochs in AMM and BSGD (%d)\n", (*param).NUM_EPOCHS);
		mexPrintf("\t s - number of subepochs in AMM batch (%d)\n", (*param).NUM_SUBEPOCHS);
		mexPrintf("\t b - bias term in AMM, if 0 no bias added (%.1f)\n", (*param).BIAS_TERM);
		mexPrintf("\t k - pruning frequency, after how many observed examples is pruning done in AMM (%d)\n", (*param).K_PARAM);
		mexPrintf("\t C - pruning aggresiveness, sets the pruning threshold in AMM, OR\n");
		mexPrintf("\t\t     linear-SVM regularization paramater C in LLSVM (%.1f)\n", (*param).C_PARAM);
		mexPrintf("\t l - limit on the number of weights per class in AMM (%d)\n", (*param).LIMIT_NUM_WEIGHTS_PER_CLASS);
		mexPrintf("\t L - learning parameter lambda in AMM and BSGD (%.5f)\n", (*param).LAMBDA_PARAM);
		mexPrintf("\t G - kernel width exp(-0.5 * gamma * ||x_i - x_j||^2) in BSGD and LLSVM (1 / DIMENSIONALITY)\n");
		mexPrintf("\t B - total SV set budget in BSGD, OR number of landmard points in LLSVM (%d)\n", (*param).BUDGET_SIZE);
		mexPrintf("\t M - budget maintenance strategy in BSGD (0 - removal; 1 - merging), OR\n");
		mexPrintf("\t\t     landmark sampling strategy in LLSVM (0 - random; 1 - k-means; 2 - k-medoids) (%d)\n\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
		
		mexPrintf("\t z - training and test file are loaded in chunks so that the algorithm can \n");
		mexPrintf("\t\t     handle budget files on weaker computers; z specifies number of examples loaded in\n");
		mexPrintf("\t\t     a single chunk of data, ONLY when inputs are .txt files (%d)\n", (*param).CHUNK_SIZE);
		mexPrintf("\t w - model weights are split in chunks, so that the algorithm can handle\n");
		mexPrintf("\t\t     highly dimensional data on weaker computers; w specifies number of dimensions stored\n");
		mexPrintf("\t\t     in one chunk, ONLY when inputs are .txt files (%d)\n", (*param).CHUNK_WEIGHT);
		mexPrintf("\t S - if set to 1 data is assumed sparse, if 0 data is assumed non-sparse, used to\n");
		mexPrintf("\t\t     speed up kernel computations (default is 1 when percentage of non-zero\n");
		mexPrintf("\t\t     features is less than 5%%, and 0 when percentage is larger than 5%%)\n");
		mexPrintf("\t v - verbose output: 1 to show the algorithm steps (epoch ended, training started, ...), 0 for quiet mode (%d)\n", (*param).VERBOSE);
		mexPrintf("\t--------------------------------------------\n");
		mexPrintf("\tInstructions on how to convert data to and from the LIBSVM format can be found on <a href=\"http://www.csie.ntu.edu.tw/~cjlin/libsvm/\">LIBSVM website</a>.\n");
	}
	else
	{
		mexPrintf("\n\tUsage:\n");
		mexPrintf("\t\t[error_rate, pred_labels] = budgetedsvm_predict(label_vector, instance_matrix, model, parameter_string = '')\n\n");
		mexPrintf("\tInputs:\n");
		mexPrintf("\t\tlabel_vector\t\t- label vector of size (N x 1), a label set can include any integer\n");
		mexPrintf("\t\t\t\t\t              representing a class, such as 0/1 or +1/-1 in the case of binary-class\n");
		mexPrintf("\t\t\t\t\t              problems; in the case of multi-class problems it can be any set of integers\n");
		mexPrintf("\t\tinstance_matrix\t\t- instance matrix of size (N x DIMENSIONALITY),\n");
		mexPrintf("\t\t\t\t                  where each row represents one example\n");
		mexPrintf("\t\tmodel\t\t\t\t- structure holding the model learned through budgetedsvm_train()\n");
		mexPrintf("\t\tparameter_string\t- parameters of the model, defaults to empty string if not provided\n\n");
		mexPrintf("\tOutput:\n");
		mexPrintf("\t\terror_rate\t\t\t- error rate on the test set\n");
		mexPrintf("\t\tpred_labels\t\t\t- vector of predicted labels of size (N x 1)\n\n");		
		mexPrintf("\t--------------------------------------------\n\n");
		
		mexPrintf("\tIf the data set cannot be fully loaded to Matlab, another variant can be used:\n");
		mexPrintf("\t\t[error_rate, pred_labels] = budgetedsvm_predict(test_file, model_file, parameter_string = '')\n\n");
		mexPrintf("\tInputs:\n");
		mexPrintf("\t\ttest_file\t\t\t- filename of .txt file containing test data set in LIBSVM format\n");
		mexPrintf("\t\tmodel_file\t\t\t- filename of .txt file containing model trained through budgetedsvm_train()\n");
		mexPrintf("\t\tparameter_string\t- parameters of the model, defaults to empty string if not provided\n\n");
		mexPrintf("\tOutput:\n");
		mexPrintf("\t\terror_rate\t\t\t- error rate on the test set\n");
		mexPrintf("\t\tpred_labels\t\t\t- vector of predicted labels of size (N x 1)\n\n");
		
		mexPrintf("\t--------------------------------------------\n\n");
		mexPrintf("\tParameter string is of the following format:\n");
		mexPrintf("\t'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'\n\n");
		mexPrintf("\tThe following options are available (default values in parentheses):\n");
		mexPrintf("\tz - the training and test file are loaded in chunks so that the algorithm can\n");
		mexPrintf("\t\t    handle budget files on weaker computers; z specifies number of examples loaded in\n");
		mexPrintf("\t\t    a single chunk of data, ONLY when inputs are .txt files (%d)\n", (*param).CHUNK_SIZE);
		mexPrintf("\tw - the model weight is split in parts, so that the algorithm can handle\n");
		mexPrintf("\t\t    highly dimensional data on weaker computers; w specifies number of dimensions stored\n");
		mexPrintf("\t\t    in one chunk, ONLY when inputs are .txt files (%d)\n", (*param).CHUNK_WEIGHT);
		mexPrintf("\tS - if set to 1 data is assumed sparse, if 0 data is assumed non-sparse, used to\n");
		mexPrintf("\t\t    speed up kernel computations (default is 1 when percentage of non-zero\n");
		mexPrintf("\t\t    features is less than 5%%, and 0 when percentage is larger than 5%%)\n");
		mexPrintf("\tv - verbose output: 1 to show algorithm steps, 0 for quiet mode (%d)\n", (*param).VERBOSE);
		mexPrintf("\t--------------------------------------------\n");
		mexPrintf("\tInstructions on how to convert data to and from the LIBSVM format can be found on <a href=\"http://www.csie.ntu.edu.tw/~cjlin/libsvm/\">LIBSVM website</a>.\n");		
	}
}

/* \fn void parseInputMatlab(parameters *param, const char *paramString, bool trainingPhase, const char *inputFileName, const char *modelFileName)
	\brief Parses the user input and modifies parameter settings as necessary.
	\param [out] param Parameter object modified by user input.
	\param [in] paramString User-provided parameter string, can be NULL in which case default parameters are used.
	\param [in] trainingPhase Indicator if training or testing phase.
	\param [in] inputFileName Filename of the file that holds the data.
	\param [in] modelFileName Filename of the file that will hold the model (if trainingPhase = 1), or that holds the model (if trainingPhase = 0).
*/
void parseInputMatlab(parameters *param, const char *paramString, bool trainingPhase, const char *inputFileName, const char *modelFileName)
{	
	int pos = 0, tempPos = 0, len;
	char str[256];
	vector <char> option;
	vector <float> value;
	FILE *pFile = NULL;
	
	if (paramString == NULL)
		len = 0;
	else
		len = strlen(paramString);
	
	// check if the input data file exists only if input data filename is provided
	if (inputFileName)
	{
		if (!readableFileExists(inputFileName))
		{
			sprintf(str, "Can't open input file %s!\n", inputFileName);
			mexErrMsgTxt(str);
		}
	}
	
	while (pos < len)
	{
		if (paramString[pos++] == '-')
		{
			option.push_back(paramString[pos]);
			pos += 2;

			tempPos = 0;
			while ((paramString[pos] != ' ') && (paramString[pos] != '\0'))
			{
				str[tempPos++] = paramString[pos++];
			}
			str[tempPos++] = '\0';
			value.push_back((float) atof(str));
		}
	}
		
	if (trainingPhase)
	{
		// check if the model file exists only if model filename is provided
		if (modelFileName)
		{
			pFile = fopen(modelFileName, "w");
			if (pFile == NULL)
			{
				sprintf(str, "Can't create model file %s!\n", modelFileName);
				mexErrMsgTxt(str);
			}
			else
			{
				fclose(pFile);
				pFile = NULL;
			}
		}
		
		// modify parameters
		for (unsigned int i = 0; i < option.size(); i++)
		{
			switch (option[i])
			{
				case 'A':
					(*param).ALGORITHM = (unsigned int) value[i];
					if ((*param).ALGORITHM > 4)
					{
						sprintf(str, "Input parameter '-A %d' out of bounds!\nRun 'budgetedsvm_train()' for help.", (*param).ALGORITHM);
						mexErrMsgTxt(str);
					}
					break;
				case 'e':
					(*param).NUM_EPOCHS = (unsigned int) value[i];
					break;
				case 's':
					(*param).NUM_SUBEPOCHS = (unsigned int) value[i];
					break;
				case 'k':
					(*param).K_PARAM = (unsigned int) value[i];
					break;
				case 'C':
					(*param).C_PARAM = value[i];
					if ((*param).C_PARAM < 0.0)
					{
						sprintf(str, "Input parameter '-C' should be a positive real number!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'L':
					(*param).LAMBDA_PARAM = (double) value[i];
					if ((*param).LAMBDA_PARAM < 0.0)
					{
						sprintf(str, "Input parameter '-L' should be a positive real number!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				
				// these three are for the BSGD
				case 'B':
					(*param).BUDGET_SIZE = (unsigned int) value[i];
					break;
				case 'G':
					(*param).GAMMA_PARAM = (long double) value[i];
					if ((*param).GAMMA_PARAM < 0.0)
					{
						sprintf(str, "Input parameter '-G' should be a positive real number!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'M':
					(*param).MAINTENANCE_SAMPLING_STRATEGY = (unsigned int)  value[i];
					break;  
				
				case 'b':
					(*param).BIAS_TERM = (double) value[i];
					break;
				case 'v':
					(*param).VERBOSE = (value[i] != 0);
					break;
				case 'l':
					(*param).LIMIT_NUM_WEIGHTS_PER_CLASS = (unsigned int) value[i];
					if ((*param).LIMIT_NUM_WEIGHTS_PER_CLASS < 1)
					{
						sprintf(str, "Input parameter '-l' should be an integer larger than 0!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'd':
					// a user explicitly assigns dimensionality only if data set is given in .txt file, otherwise dimensionality is found directly from Matlab, no need for a user to specify it
					if (inputFileName)
					{
						(*param).DIMENSION = (unsigned int) value[i];
					}
					else
					{
						sprintf(str, "Error, unknown input parameter '-d'!\nRun 'budgetedsvm_train()' for help.", option[i]);
						mexErrMsgTxt(str);
					}
					break;				
				
				case 'z':
					(*param).CHUNK_SIZE = (unsigned int) value[i];
					if ((*param).CHUNK_SIZE < 1)
					{
						sprintf(str, "Input parameter '-z' should be an integer larger than 0!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'w':
					(*param).CHUNK_WEIGHT = (unsigned int) value[i];
					if ((*param).CHUNK_WEIGHT < 1)
					{
						sprintf(str, "Input parameter '-w' should be an integer larger than 0!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'S':
					(*param).VERY_SPARSE_DATA = (unsigned int) (value[i] != 0);
					break;

				default:
					sprintf(str, "Error, unknown input parameter '-%c'!\nRun 'budgetedsvm_train()' for help.", option[i]);
					mexErrMsgTxt(str);
					break;
			}
		}
		
		// check the MAINTENANCE_SAMPLING_STRATEGY validity
		if ((*param).ALGORITHM == LLSVM)
		{
			if ((*param).MAINTENANCE_SAMPLING_STRATEGY > 2)
			{
				// 0 - random removal, 1 - k-means, 2 - k-medoids
				sprintf(str, "Error, unknown input parameter '-M %d'!\nRun 'budgetedsvm-train' for help.\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
				svmPrintErrorString(str);
			}
		}
		else if ((*param).ALGORITHM == BSGD)
		{
			if ((*param).MAINTENANCE_SAMPLING_STRATEGY > 1)
			{
				// 0 - smallest removal, 1 - merging
				sprintf(str, "Error, unknown input parameter '-M %d'!\nRun 'budgetedsvm-train' for help.\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
				svmPrintErrorString(str);
			}
		}
		
		if ((*param).VERBOSE)
		{
			mexPrintf("*** Training started with the following parameters:\n");
			switch ((*param).ALGORITHM)
			{
				case PEGASOS:
					mexPrintf("Algorithm \t\t\t\t: Pegasos\n");
					break;
				case AMM_ONLINE:
					mexPrintf("Algorithm \t\t\t\t: AMM online\n");
					break;
				case AMM_BATCH:
					mexPrintf("Algorithm \t\t\t\t: AMM batch\n");
					break;
				case BSGD:
					mexPrintf("Algorithm \t\t\t\t: BSGD\n");
					break;
				case LLSVM:
					mexPrintf("Algorithm \t\t\t\t\t: LLSVM\n");
					break;
			}
			
			if (((*param).ALGORITHM == PEGASOS) || ((*param).ALGORITHM == AMM_BATCH) || ((*param).ALGORITHM == AMM_ONLINE))
			{
				mexPrintf("Lambda parameter \t\t: %f\n", (*param).LAMBDA_PARAM);
				mexPrintf("Bias term \t\t\t\t: %f\n", (*param).BIAS_TERM);
				if ((*param).ALGORITHM != PEGASOS)
				{
					mexPrintf("Pruning frequency k \t: %d\n", (*param).K_PARAM);
					mexPrintf("Pruning threshold C \t: %f\n", (*param).C_PARAM);
					mexPrintf("Num. weights per class\t: %d\n", (*param).LIMIT_NUM_WEIGHTS_PER_CLASS);
					mexPrintf("Number of epochs \t\t: %d\n\n", (*param).NUM_EPOCHS);
				}
				else
					mexPrintf("\n");
			}
			else if ((*param).ALGORITHM == BSGD)
			{
				mexPrintf("Number of epochs \t\t: %d\n", (*param).NUM_EPOCHS);
				if ((*param).MAINTENANCE_SAMPLING_STRATEGY == 0)
					mexPrintf("Maintenance strategy \t: 0 (smallest removal)\n");
				else
					mexPrintf("Maintenance strategy \t: 1 (merging)\n");
				mexPrintf("Lambda parameter \t\t: %f\n", (*param).LAMBDA_PARAM);
				if ((*param).GAMMA_PARAM != 0.0)
					mexPrintf("Gaussian kernel width \t: %f\n", (*param).GAMMA_PARAM);
				else
					mexPrintf("Gaussian kernel width \t: 1 / DIMENSIONALITY\n");
				mexPrintf("Size of the budget \t\t: %d\n\n", (*param).BUDGET_SIZE);
			}
			else if ((*param).ALGORITHM == LLSVM)
			{
				switch ((*param).MAINTENANCE_SAMPLING_STRATEGY)
				{
					case 0:
						mexPrintf("Landmark sampling \t\t\t: 0 (random sampling)\n");
						break;
						
					case 1:
						mexPrintf("Landmark sampling \t\t\t: 1 (k-means initialization)\n");
						break;
						
					case 2:
						mexPrintf("Landmark sampling \t\t\t: 2 (k-medoids initialization)\n");
						break;
				}
				mexPrintf("Number of landmark points \t: %d\n", (*param).BUDGET_SIZE);
				mexPrintf("C regularization parameter \t: %f\n", (*param).C_PARAM);
				if ((*param).GAMMA_PARAM != 0.0)
					mexPrintf("Gaussian kernel width \t\t: %f\n\n", (*param).GAMMA_PARAM);
				else
					mexPrintf("Gaussian kernel width \t\t: 1 / DIMENSIONALITY\n\n");
			}
			mexEvalString("drawnow;");
		}
		
		// no bias term for LLSVM and BSGD functions
		if (((*param).ALGORITHM == LLSVM) || ((*param).ALGORITHM == BSGD))
			(*param).BIAS_TERM = 0.0;
		
		// if inputs to training phase are .txt files, then also increase dimensionality due to added bias term, and update GAMMA_PARAM if not set by a user;
		//	NOTE that we do not execute this part if inputs are Matlab variables, as we still do not know the dimensionality, therefore BIAS_TERM and
		//	GAMMA_PARAM are adjusted in budgetedDataMatlab::readDataFromMatlab() function, after we find out the dimensionality of the considered data set
		if (inputFileName)
		{
			// increase dimensionality if bias term included
			if ((*param).BIAS_TERM != 0.0)
			{
				(*param).DIMENSION++;
			}
			
			// set gamma to default value of dimensionality
			if ((*param).GAMMA_PARAM == 0.0)
				(*param).GAMMA_PARAM = 1.0 / (double) (*param).DIMENSION;
		} 
	}
	else
	{
		// check if the model file exists only if model filename is provided
		if (modelFileName)
		{
			if (!readableFileExists(modelFileName))
			{
				sprintf(str, "Can't open model file %s!\n", modelFileName);
				mexErrMsgTxt(str);
			}
		}
		
		// modify parameters
		for (unsigned int i = 0; i < option.size(); i++)
		{
			switch (option[i])
			{
				/*case 'p':
					(*param).SAVE_PREDS = (value[i] != 0);
					break;*/
				case 'v':
					(*param).VERBOSE = (value[i] != 0);
					break;			
				
				case 'z':
					(*param).CHUNK_SIZE = (unsigned int) value[i];					
					if ((*param).CHUNK_SIZE < 1)
					{
						sprintf(str, "Input parameter '-z' should be an integer larger than 0!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'w':
					(*param).CHUNK_WEIGHT = (unsigned int) value[i];
					if ((*param).CHUNK_WEIGHT < 1)
					{
						sprintf(str, "Input parameter '-w' should be an integer larger than 0!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'S':
					(*param).VERY_SPARSE_DATA = (unsigned int) (value[i] != 0);
					break;

				default:
					sprintf(str, "Error, unknown input parameter '-%c'!\nRun 'budgetedsvm_predict()' for help.", option[i]);
					mexErrMsgTxt(str);
					break;
			}
		}
		
		/*if ((*param).VERBOSE)
		{
			mexPrintf("\n*** Testing with the following parameters:\n");
			switch ((*param).ALGORITHM)
			{
				case PEGASOS:
					mexPrintf("Algorithm: \t\t\t\tPEGASOS\n");
					break;
				case AMM_ONLINE:
					mexPrintf("Algorithm: \t\t\t\tAMM online\n");
					break;
				case AMM_BATCH:
					mexPrintf("Algorithm: \t\t\t\tAMM batch\n");
					break;
				case BSGD:
					mexPrintf("Algorithm: \t\t\t\tBSGD\n");
					break;
			}
			
			if (((*param).ALGORITHM == PEGASOS) || ((*param).ALGORITHM == AMM_BATCH) || ((*param).ALGORITHM == AMM_ONLINE))
			{
				mexPrintf("Bias term: \t\t\t\t%f\n\n", (*param).BIAS_TERM);
			}
			else if ((*param).ALGORITHM == BSGD)
			{
				mexPrintf("Gaussian kernel width: \t%f\n\n", (*param).GAMMA_PARAM);	
			}
			mexEvalString("drawnow;");
		}*/
	}
	
	setPrintErrorStringFunction(&printErrorStringMatlab);	
	if ((*param).VERBOSE)
		setPrintStringFunction(&printStringMatlab);
	else
		setPrintStringFunction(NULL);
}