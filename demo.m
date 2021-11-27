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
clc; clear all;
srcStr = {'caltech','caltech','caltech','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgtStr = {'amazon','webcam','dslr','caltech','webcam','dslr','caltech','amazon','dslr','caltech','amazon','webcam'};
finalResult=[];
finalMu=[];
finalZs=[];
finalZt=[];
res=[];
options.p = 200;
options.lambda = 1; %1.0;
options.kernel_type = 'linear';
options.mu = 0.1;
options.iter = 20;
options.gamma = 1;
dim=10;
for i = 1:12
    src = char(srcStr{i});
    tgt = char(tgtStr{i});
    fprintf('%d: %s_vs_%s\n',i,src,tgt);
    load(['./data/' src '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xs =zscore(fts,1);
    Ys = labels;
    load(['./data/' tgt '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xt = zscore(fts,1);
    Yt = labels;
    Xs=Xs';
    Xt=Xt';
    [result,mu]=ICSC(Xs,Ys,Xt,Yt,options);
    finalResult=[finalResult;result];
    finalMu=[finalMu;mu];
end

