function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum = 0;
X_trans = X';
delta = 0;
delta_reg = zeros(size(theta)-1, 1);

for i = 1:m
  h = sigmoid(X(i,:)*theta);
  sum += -y(i)*log(h) - (1-y(i))*log(1-h);
  delta += (h-y(i))*X(i,1);   % X(i,1) shoule be 1 anyway
  delta_reg += (h-y(i)) * X_trans(2:end,i);
end

theta_del = theta(2:end,:);
J = 1/m * sum + lambda/(2*m) * theta_del' * theta_del;

grad(1) = 1/m * delta;
grad(2:end) = 1/m * delta_reg + lambda/m * theta_del;

% =============================================================
end