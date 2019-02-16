function [J, grad] = costFunction(theta, X, y)
%this function computes the cost of selecting theta as parametr for logistic regression.

%m= number of traing examples
m = length(y); 

%
temp1 = -1 * (y .* log(sigmoid(X * theta))); 
temp2 = (1 - y) .* log(1 - sigmoid(X * theta)); 
 
J = sum(temp1 - temp2) / m;



grad = (X' * (sigmoid(X * theta) - y)) * (1/m);

end