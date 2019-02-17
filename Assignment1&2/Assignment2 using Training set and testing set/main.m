clear all 
close all
clc

d = importdata('heart_DD.csv');

X=d.data(:,1:13);
Y=d.data(:,14);

[m,n] = size(X);

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


[nsamples, nfeatures] = size(trainset)
w0 = rand(nfeatures + 1, 1);
weight = logisticRegressionWeights( trainset, y_train, w0, 500, 0.1);
res = logisticRegressionClassify( test, weight );
errors = abs(y_test - res);
err = sum(errors)
percentage = 1 - err / size(trainset, 1)
