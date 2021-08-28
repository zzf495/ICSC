function mu = ACDA(Xs,Xt,Ys,PseudoYt,C)
    %% input:
    %%% Xs: source samples, ns * m;
    %%% Xt: target samples, nt * m;
    %%% Ys: source labels, ns * 1;
    %%% PseudoYt: target pseudo labels, nt *1;
    %% output:
    %%% mu: adaptive adjustment parameter
    X=[Xs;Xt];
    Y=[Ys;PseudoYt];
    Dw=0;
    class_number=0;
    % solving (10)
    for i = 1:C
        Xc=X(Y==i,:);
        Dw=Dw+norm(cov(Xc),'fro')*(length(Xc));
        class_number=class_number+length(Xc);
    end
    Dw=Dw/class_number;
    % solving (11)
    Db=norm(cov(X),'fro');
    % solving (12)
    mu = exp(-Dw/Db);
end

