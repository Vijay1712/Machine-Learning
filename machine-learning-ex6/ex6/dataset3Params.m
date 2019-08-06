function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_Test     = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_Test = [0.01 0.03 0.1 0.3 1 3 10 30]';
prediction_Error = zeros(length(C_Test), length(sigma_Test));
result = zeros(length(C_Test)+length(sigma_Test),3);
row = 1;
  
for i = 1:length(C_Test)
  for j = 1: length(sigma_Test)
    C_test = C_Test(i);
    sigma_test = sigma_Test(j);
    model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    predictions = svmPredict(model, Xval);
    prediction_Error(i,j) = mean(double(predictions ~= yval));
    result(row,:) = [prediction_Error(i,j), C_test, sigma_test];
    row = row + 1;
      end
  end
  
sorted_Result = sortrows(result, 1);
C = sorted_Result(1,2);
sigma = sorted_Result(1,3);

% =========================================================================

end
