clear all 
close all
clc
%Hypothesis Two using more features and test set no regularization

d = importdata('heart_DD.csv');

X=d.data(:,1:13);
Y=d.data(:,14);
[m,n] = size(X);
initialTheta = zeros(n+1, 1);

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

Alpha=0.5;
[m,n] = size(test);
testtheta = zeros(n+1, 1);
X= [ones(m,1) test];
s=zeros(size(X*testtheta));
h=X*testtheta;
s=1./(1+exp(-1*h));
a= -1*(y_test.* log(s));
b=(1-y_test) .* log(1-s);

%Cost Calculated
cost = sum(a-b)/m;

%gradient calculated
gradient = (X' * (s - y_test))*(1/m);

%Cross
Alpha=0.5;
[m,n] = size(cross);
crosstheta = zeros(n+1, 1);
X2= [ones(m,1) cross];
R=1;
k=1;
while R==1
crosstheta=crosstheta-(Alpha/m)*X2'*(X2*crosstheta-y_cross);
k=k+1;
E(k)=(1/(2*m))*sum((X*crosstheta-y_cross).^2);
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.0001;
    R=0;
end
end

%find accuracy of training set
z=size(X,1);
p=zeros(z,1);
result = sigmoid(X * crosstheta); 
p = round(result); 

fprintf('Training set Accuracy: %f\n', mean(double(p == y_test)) * 100);





