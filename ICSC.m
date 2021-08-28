function [result,mu]= ICSC(Xs,Ys,Xt,Yt,options)
%% Implementation of ICSC
%%% Authors: Teng et al.
%%% Doi: 10.1002/int.22629
%%% Paper: 2021-Domain Adaptation via Incremental Confidence Samples into Classification
%%% Implementation author:      ZeFeng Zheng, github: https://github.com/zzf495
%% input
%%% Xs: source samples, ns * m;
%%% Xt: target samples, nt * m;
%%% Ys: source labels, ns * 1;
%%% Yt: target labels (only use for calculating accuracy), nt *1;
%%% options.iter: the number of iterations;
%%% options.p: the number of dimension;
%% output
%%% result:     the classification accuracy (list)
%%% mu:         the adjusted parameter (list)
mu=[];
if ~isfield(options,'iter')
    options.iter = 10;
end
if ~isfield(options,'p')
    options.p = size(Xs,1);
end
dim=options.p;
if dim>size(Xs,1)
     options.p = size(Xs,1);
end
T = options.iter;
C = length(unique(Ys));
result=[];
pseudoLabels=[];
for iter = 1:T
    % solving (9)
    [Zs,Zt] = JPDA(Xs,Xt,Ys,pseudoLabels,options);
    % solving (5)
    Zmean = mean([Zs;Zt]);
    Zs = Zs - repmat(Zmean,[size(Zs,1) 1 ]);
    Zt = Zt - repmat(Zmean,[size(Zt,1) 1 ]);
    Zs = L2Norm(Zs);
    Zt = L2Norm(Zt);
    %% distance to class means
    classMeans = zeros(C,size(Zs,2));
    for i = 1:C
        classMeans(i,:) = mean(Zs(Ys==i,:));
    end
    % solving (6)
    classMeans = L2Norm(classMeans);
    distClassMeans = EuDist2(Zt,classMeans);
    expMatrix = exp(-distClassMeans);
    probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 C]);
    % solving (7)
    [prob,predLabels] = max(probMatrix');
    % Definition 1
    selelctPercet=iter/T;
    p=(1-selelctPercet);   %*(1-options.selective);
    [sortedProb,index] = sort(prob);
    sortedPredLabels = predLabels(index);
    trustable = zeros(1,length(prob));
    for i = 1:C
        thisClassProb = sortedProb(sortedPredLabels==i);
        if ~isempty(thisClassProb)
            trustable = trustable+ (prob>thisClassProb(floor(length(thisClassProb)*p)+1)).*(predLabels==i);
        end
    end
    % Definition 2
    pseudoLabels = predLabels;
    pseudoLabels(~trustable) = -1;
    % calculate ACC
    acc = sum(predLabels'==Yt)/length(Yt);
    result=[result,acc];
    % solving (10)-(12)
    options.mu = ACDA(Zs,Zt,Ys,pseudoLabels',C);
    mu=[mu,options.mu];
    fprintf('Iteration=%d,mu=%0.3f, Acc:%0.3f\n', iter,options.mu, acc);
end
end



function [Zs,Zt] = JPDA(Xs,Xt,Ys,YtPseudo,options)
%% source code: https://github.com/chamwen/JPDA
%%% Joint Probability Distribution Adaptation (JPDA)
%%% Author: Wen Zhang
%%% Date: Dec. 8, 2019
%%% E-mail: wenz@hust.edu.cn
p = options.p;
lambda = options.lambda;
ker = options.ker;
mu = options.mu;
gamma = options.gamma;

% Set variables
X = [Xs,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
C = length(unique(Ys));

% solving (16)
Ns=1/ns*onehot(Ys,unique(Ys)); Nt=zeros(nt,C);
if ~isempty(YtPseudo); Nt=1/nt*onehot(YtPseudo,unique(Ys)); end

Rmin=[Ns*Ns',-Ns*Nt';-Nt*Ns',Nt*Nt'];
Rmin = Rmin / norm(Rmin,'fro');

% solving (17)
Ms=[]; Mt=[];
for i=1:C
    Ms=[Ms,repmat(Ns(:,i),1,C-1)];
    idx=1:C; idx(i)=[];
    Mt=[Mt,Nt(:,idx)];
end
Rmax=[Ms*Ms',-Ms*Mt';-Mt*Ms',Mt*Mt'];
Rmax = Rmax / norm(Rmax,'fro');
H = eye(n)-1/(n)*ones(n,n);
if strcmp(ker,'primal')
    [A,~] = eigs(X*((1-mu)*Rmin-mu*Rmax)*X'+lambda*eye(m),X*H*X',p,'SM');
    Z = A'*X;
else
    K = kernel(ker,X,[],gamma);
    [A,~] = eigs(K*((1-mu)*Rmin-mu*Rmax)*K'+lambda*eye(n),K*H*K',p,'SM');
    Z = A'*K;
end
Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
Zs = Z(:,1:size(Xs,2));
Zt = Z(:,size(Xs,2)+1:end);
Zs=Zs';
Zt=Zt';
end

function y_onehot=onehot(y,class)
% Encode label to onehot form
% Input:
% y: label vector, N*1
% Output:
% y_onehot: onehot label matrix, N*C
nc=length(class);
y_onehot=zeros(length(y), nc);
for i=1:length(y)
    y_onehot(i, class==y(i))=1;
end
end

function K = kernel(ker,X,X2,gamma)
% With Fast Computation of the RBF kernel matrix
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix
%
% Gustavo Camps-Valls
% 2006(c)
% Jordi (jordi@uv.es), 2007
switch ker
    case 'linear'
        
        if isempty(X2)
            K = X'*X;
        else
            K = X'*X2;
        end
        
    case 'rbf'
        
        n1sq = sum(X.^2,1);
        n1 = size(X,2);
        
        if isempty(X2)
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*(X'*X);
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end
        K = exp(-gamma*D);
        
    case 'sam'
        
        if isempty(X2)
            D = X'*X;
        else
            D = X'*X2;
        end
        K = exp(-gamma*acos(D).^2);
        
    otherwise
        error(['Unsupported kernel ' ker])
end
end