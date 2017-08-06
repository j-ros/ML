function [error_train, error_val] = ...
    learningCurve_rand(X, y, Xval, yval, lambda, rep)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve averaging for rep repetitions each.
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda, rep) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)) over rep repetitions
%


% Number of training examples
m = size(X, 1);
% Number of validation examples
m2 = size(Xval, 1);

% Initialize output values
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m,
    error_train_acum = zeros(rep, 1);
    error_val_acum = zeros(rep, 1);
  for j = 1:rep,
    %select i examples from each set with replacement
    idx_train = randi(m, i, 1); 
    idx_val = randi(m2, i, 1); 
    X_sel = X(idx_train, :);
    y_sel = y(idx_train, :);
    Xval_sel = Xval(idx_val, :); 
    yval_sel = yval(idx_val, :);
    theta = trainLinearReg(X_sel, y_sel, lambda);
    error_train_acum( j ) = linearRegCostFunction(X_sel, y_sel, theta, 0);
    error_val_acum( j ) = linearRegCostFunction(Xval_sel, yval_sel, theta, 0);
   end;
   error_train(i) = mean(error_train_acum);
   error_val(i)   = mean(error_val_acum);
end;

end
