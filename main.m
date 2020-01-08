%% Initalization and load data
clc;
load("linear_svm.mat");
%% Visulize the train dataset
% figure(1);
index1 = find(labels_train == 1);
index2 = find(labels_train == -1);
X1 = X_train(index1,:);
X2 = X_train(index2,:);
scatter(X1(:,1),X1(:,2), 'b');
hold on;
scatter(X2(:,1),X2(:,2), 'r');
title("Visulizatio of the train set");
grid on;
%% Dual domain solution
% Build G matrix 
N = length(X_train);
G = zeros(N,N);
for i = 1:N
    for j = 1:N
        G(i,j) = labels_train(i) * labels_train(j) * X_train(j,:)*X_train(i,:)'; 
    end
end
u = ones(N,1);
%% call matlab solution for this dual problem
[alpha,fval] = quadprog(G,-u,-eye(N),zeros(N,1),labels_train',0);
% interior-point method
% the method is complex and it would be too much effort implement it here.
% Go to the source code for details
w = sum(alpha .* labels_train .* X_train);
b = -0.5 * (max(X2 * w') + min(X1 * w'));
m = -w(1)/w(2);
c = -b/w(2);
figure(1);
fplot(@(x) m*x+c,[1.5,2.4],'k');
legend("Label = 1","Label = -1","classifier - quadprog");
%% Try solve this within in primal domain with Hinge loss function
% Hingle loss max(0; 1 ? yi(wT xi + b))
% start a simple gradient decent to solve it
wp = [1;1];
lambda = 1;
bp = 0;
err = 1e-5;
eta = 0.0001;
y = labels_train;
x = X_train;
k = 1;
iterations = 30000;
fixstep = 0.5;
while(k < iterations)
errorset = y .*(x * wp + bp);
Correctindex = find(errorset > 1);

% Determine the gradient for w
% it is an average for all the data points contributes to the change
% direction is not contributed by those have been properly done

gradW = zeros(N,2);
gradW(:,1) = -y.*x(:,1);
gradW(:,2) = -y.*x(:,2);
gradW(Correctindex,:) = 0;
gradw = mean(gradW);

% Determine the gradient for b
gradB = -y;
gradB(Correctindex) = 0;
gradb = mean(gradB);
if abs(gradw(1)) < err  && abs(gradw(2)) < err && abs(gradb) < err
    break;
end
% Iterate the (w,b)
wp = wp - fixstep * gradw';
bp = bp - fixstep * gradb;
k = k + 1;
end
figure(1);
m = -wp(1)/wp(2);
c = -bp/wp(2);
fplot(@(x) m*x+c,[1.5,2.4],'y');
legend("Label = 1","Label = -1","classifier - quadprog","claaifier Hinge loss - GD");
