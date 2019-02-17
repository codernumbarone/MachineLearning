function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); 

J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
J = (-y'*log(h) - (1-y)'*log(1-h))/m + sum(theta(2:length(theta)).^2)*(lambda/(2*m));
grad = (X'*(h-y))/m + (lambda*theta)/m;
grad(1) = grad(1)-(lambda*theta(1))/m;
% =============================================================
end
