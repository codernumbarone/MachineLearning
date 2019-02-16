clear all 
close all
clc
%Hypothesis One

d = importdata('heart_DD.csv');

X=d.data(:,[4,5]);
Y=d.data(:,14);

Alpha=0.5;
[m,n] = size(X);
X= [ones(m,1) X];

initialTheta = zeros(n+1, 1);
s=zeros(size(X*initialTheta));
h=X*initialTheta;
s=1./(1+exp(-1*h));
a= -1*(Y.* log(s));
b=(1-Y) .* log(1-s);

%Cost Calculated
cost = sum(a-b)/m;

%gradient calculated
gradient = (X' * (s - Y))*(1/m);


opt = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = fminunc(@(t)(costFunction2(t, X, Y)),initialTheta, opt);

%find accuracy of training set
z=size(X,1);
p=zeros(z,1);
result = sigmoid(X * theta); 
p = round(result); 

fprintf('Training set Accuracy: %f\n', mean(double(p == Y)) * 100);




