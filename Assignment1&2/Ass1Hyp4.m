clear all

%Hyphothesis four
ds = tabularTextDatastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
size(T);
Alpha=0.5;
m=length(T{:,1});
U0=T{:,2}
U=T{:,4:19};
U1=T{:,20:21};
AllU=[U U1];
X=[ones(m,1) U U1];

n=length(X(1,:));
for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
end

Y=T{:,3}/mean(T{:,3});
y=length(Y);
Theta=zeros(n,1);
k=1;

E(k)=(1/(2*m))*sum((X*Theta-Y).^2);

R=1;
while R==1
Theta=Theta-(Alpha/m)*X'*(X*Theta-Y);
k=k+1;
E(k)=(1/(2*m))*sum((X*Theta-Y).^2);
if E(k-1)-E(k)<0
    Alpha=Alpha*0.5;
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.001;
    R=0;
end
end

%Normal Equation
Z = [ones(y, 1) AllU];
th = zeros(size(Z, 3), 1);
th = pinv((Z'*Z))*Z'*Y;

fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', th);
fprintf('\n');
