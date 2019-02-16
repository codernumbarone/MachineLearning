function [cost, grad] = costFunction(theta, X, y)
%this function computes the cost of selecting theta as parametr for logistic regression.

%m= number of traing examples
m= length(y);
s=zeros(size(X*theta));
h=X*initialTheta;
s=1./(1+exp(-1*h));
a= -1*(y.* log(s));
b=(1-y) .* log(1-s);

%Cost Calculated
cost = sum(a-b)/m;

%gradient calculated
grad = (X' * (s - y))*(1/m);

end