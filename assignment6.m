clear all
close all 
clc
M = importdata('house_prices_data_training_data.csv');
x=M.data(:,2:19);
[m, n] = size(x);
Y=M.data(:,1);

%Correlation
Corr_x = corr(x);
%Covariance
x_cov=cov(x) ;
%principal components 
[U S V] = svd(x_cov);
A = diag(S);

for k=1:length(A)
    a=1-(sum(A(1:k))/sum(A));
    if(a<=0.001)
        break;
    end
end
%Reduced data
R = U(:,1:k)'*x';
%Approx Data
aprox = R'*U(1:k,1:k);

Error = 1/length(A) * sum(aprox-R');

% Theta=zeros(n,1);
% k=1;
% E(k)=(1/(2*m))*sum((X*Theta-Y).^2);
% R=1;
% while R==1
% Alpha=Alpha*1;
% Theta=Theta-(Alpha/m)*X'*(X*Theta-Y);
% k=k+1;
% E(k)=(1/(2*m))*sum((X*Theta-Y).^2);
% if E(k-1)-E(k)<0
%     break
% end 
% q=(E(k-1)-E(k))./E(k-1);
% if q <.001;
%     R=0;
% end
% end


% K Means
centroids = zeros(k, size(x, 2));
randdis = randperm(size(x, 1));
centroids = x(randdis(1:k), :);
K = size(centroids, 1);
iterations = 10;
closestindex = zeros(size(x,1), 1);


for i = 1:length(closestindex)
    distance = zeros(K, 1);
    for j = 1:K
        distance(j) = sum(sum((x(i, :) - centroids(j, :)) .^ 2 ));
    end
    [closest_distance, closestindex(i)] = min(distance);
end
centroids = zeros(K, n);
for i=1:K
  indexes = find(closestindex == i);
  if size(indexes, 1) > 0
    centroids(i, :) = mean(x(indexes, :));
  end
end
%Kmean on reduced
centroids1 = zeros(k, size(R, 2));
randdis1 = randperm(size(R, 1));
centroids1 = R(randdis1(1:k), :);
K1 = size(centroids1, 1);
closestindex = zeros(size(R,1), 1);


for i = 1:length(closestindex)
    distance = zeros(K, 1);
    for j = 1:K1
        distance(j) = sum(sum((R(i, :) - centroids1(j, :)) .^ 2 ));
    end
    [closest_distance, closestindex(i)] = min(distance);
end
centroids1 = zeros(K1, n);
for i=1:K
  indexes = find(closestindex == i);
  if size(indexes, 1) > 0
    centroids1(i, :) = mean(R(indexes, :));
  end
end


%anomly detection
mu = zeros(n, 1);
Sigma = zeros(n, 1);
%get gaussian parameters
mu = mean(x)';
sigma = var(x, 1)';
%PDF
kk = length(mu);
if (size(Sigma, 2) == 1) || (size(Sigma, 1) == 1)
    Sigma = diag(Sigma);
end

% pdf = (2*pi)^(-kk/2)*det(Sigma) ^ (-0.5)*exp(-0.5) *(x-mu)'*Sigma^-1*(x-u);

X = bsxfun(@minus, x, mu(:)');
p = (2 * pi) ^ (- kk / 2) * det(Sigma) ^ (-0.5) * ...
    exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma), X), 2));

num_array = randperm(m,m);

cross_num =     (m * 0.2); 
cross     = zeros(round(cross_num),n);
y_cross = zeros(round(cross_num),1);

for i = 1:cross_num
    cross(i,:) = X(num_array(i),:);
    y_cross(i) = Y(num_array(i));
end


%PDF XVAL
% pdf = (2*pi)^(-kk/2)*det(Sigma) ^ (-0.5)*exp(-0.5) *(x-mu)'*Sigma^-1*(x-u);

XCross = bsxfun(@minus, cross, mu(:)');
pp = (2 * pi) ^ (- kk / 2) * det(Sigma) ^ (-0.5) * ...
    exp(-0.5 * sum(bsxfun(@times, XCross * pinv(Sigma), XCross), 2));



Threshold = 0;
an = 0;
anom = 0;

stepsize = (max(p) - min(p)) / 1000;
for epsilon = min(p):stepsize:max(p)
        Prediction = (p < epsilon);
    
    tp = sum((Y==1) & (Prediction==1));
    fp = sum((Y==0) & (Prediction==1));
    fn = sum((Y==1) & (Prediction==0));
    
    prec = tp / (tp+fp);
    rec = tp / (tp+fn);
    
    anom = (2*prec*rec) / (prec+rec);
    
    if anom > an
       an = anom;
       Threshold = epsilon;
    end
end
anomily = find(p < epsilon);



