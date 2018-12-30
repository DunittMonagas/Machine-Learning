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
options = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
m = size(options, 2);

[p, q] = meshgrid(options, options);
results = [p(:) q(:)];
results = [results zeros(m * m, 1)];

for i = 1:m*m

	model = svmTrain(X, y, results(i, 1), @(x1, x2) gaussianKernel(x1, x2, results(i, 2)));
	predictions = svmPredict(model, Xval);

	results(i, 3) = mean(double(predictions ~= yval));

endfor

[minimum, idx] = min(results(:, 3));

C = results(idx, 1);
sigma = results(idx, 2);

% =========================================================================

end
