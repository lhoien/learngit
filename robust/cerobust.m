clc
%生成矩阵
r=3;
m=200;
n=200;
data0 = randn(m,r)*randn(r,n);
%添加噪声
% sigma = 0e-2;
% noise = randn(size(data0));
% level = sigma*norm(data0)/norm(noise);
% NM = level*noise;
% datan = data0 + NM; 
noise = randn(size(data0));
level = randi(m,n);
NM = level.*noise;
datan = data0 + NM; 

%采样
%  %
% CL = rand(size(data0))<0.5;
% dataz = data0.*CL;
% T2=T1;
% dataz(find(dataz==0)) = nan;
 %
% CL = rand(size(datan))<0.5;
% dataz = datan.*CL;
% dataz(find(dataz==0)) = nan;
%
D = datan;
gama = 1;
[X_hat , Y_hat , Z_hat , W_hat , iter] = robust(D, gama);
disp(rank(X_hat));
disp(norm(X_hat-data0)/norm(data0));
disp(norm(Y_hat-NM)/norm(NM));