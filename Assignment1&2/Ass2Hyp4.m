clear all 
close all
clc
%Hypothesis Four using multiple model classes
d = importdata('heart_DD.csv');

X=d.data(:,1:13);
Y=d.data(:,14);
[m,n] = size(X);
U1=X.^2;
U2=X.^3;
r=length(Y);


% Seperate the training set | crossvalidation | test set
trainset_num = (m *  0.6); 
trainset = zeros(trainset_num,n);
y_train = zeros(trainset_num,1);
cross_num =     (m * 0.2); 
cross     = zeros(cross_num,n);
y_cross = zeros(cross_num,1);
test_num =       (m * 0.2); 
test        = zeros(test_num,n);
y_test = zeros(test_num,1);

num_array = randperm(m,m);
%function randperm returns a row vector containing a random permutation of the integers from 1 to m inclusive.
for i = 1:trainset_num
    trainset(i,:) = X(num_array(i),:);
    y_train(i) = Y(num_array(i));
end
for i = 1:cross_num
    cross(i,:) = X(num_array(i+trainset_num),:);
    y_cross(i) = Y(num_array(i+trainset_num));
end
for i = 1:test_num
    test(i,:) = X(num_array(i+trainset_num+cross_num),:);
    y_test(i) = Y(num_array(i+trainset_num+cross_num));
end

lambda = 0.01;
X= [ones(m,1) X U1 U2];
nn=length(X(1,:));
initialTheta = zeros(nn, 1);
cost = 0;
grad = zeros(size(initialTheta));

h = sigmoid(X*initialTheta);
cost = (-Y'*log(h) - (1-Y)'*log(1-h))/r + sum(initialTheta(1:length(initialTheta)).^2)*(lambda/(2*r));
grad = (X'*(h-Y))/r + (lambda*initialTheta)/r;
grad(1) = grad(1)-(lambda*initialTheta(1))/r;


options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, cost] = fminunc(@(t)(costFunctionReg(t, X, Y, lambda)), initialTheta, options);

%accuracy
o=size(X,1);
p = zeros(o, 1);
p(find(sigmoid(X*theta) >= 0.5)) = 1;

fprintf('Train Accuracy: %f\n', mean(double(p == Y)) * 100);


