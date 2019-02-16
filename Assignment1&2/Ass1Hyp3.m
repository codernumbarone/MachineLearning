clear all

%Hyphothesis Three using training set
ds = tabularTextDatastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
size(T);
Alpha=0.5;

U0=T{:,2}
U=T{:,4:19};
[m,n] = size(T);
X=[ones(m,1) U];
n=length(X(1,:));


for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
end

Y=T{:,3}/mean(T{:,3});
k=1;


% Seperate the training set | crossvalidation | test set
trainset_num = round(m*0.6); 
trainset = zeros(trainset_num,n);
y_train = zeros(trainset_num,1);
cross_num =   round(m * 0.2); 
cross     = zeros(cross_num,n);
y_cross = zeros(cross_num,1);
test_num =      round(m * 0.2); 
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

y=length(y_train);
Theta=zeros(size(trainset,2),1);
m=length(trainset);

R=1;
while R==1
Theta=Theta-(Alpha/m)*trainset'*(trainset*Theta-y_train);
k=k+1;
E(k)=(1/(2*m))*sum((trainset*Theta-y_train).^2);
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.01;
    R=0;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%using cross set

Theta2=zeros(size(cross,2),1);
mm=length(cross);

kk=1;
R=1;
while R==1
Theta2=Theta2-(Alpha/mm)*cross'*(cross*Theta2-y_cross);
kk=kk+1;
EE(kk)=(1/(2*mm))*sum((cross*Theta-y_cross).^2);
if EE(kk-1)-EE(kk)<0
    break
end 
qq=(EE(kk-1)-EE(kk))./EE(kk-1);
if qq <.01;
    R=0;
end
end

%Normal Equation
Z = [ones(y, 1) trainset];
th = zeros(size(Z, 3), 1);
th = pinv((Z'*Z))*Z'*y_train;

fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', th);
fprintf('\n');


