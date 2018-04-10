function [X_hat , Y_hat , Z_hat , W_hat , iter] = robust(D, gama, tol, maxIter, E, A, B, C)

%
% This matlab code implements an Alternating Direction Method (ADM) 
% for Robust Network Compressive Sensing.
%
%Input:
% D - m x n matrix of observations/data (required input)
%
% gama - weight to capture our confidence in domain knowledge
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
%
% E - a binary error indicator matrix 
%     such that E[i,j] = 1 if and only if entry D[i,j] is erroneous or missing
% 
% A - a binary routing matrix: 
%     A(i,j) = 1 if link i is used to route traffic for the j-th end-to-end flow, 
%     and A(i,j) = 0 otherwise
%
% B - an over-complete anomaly profile matrix.we can simply set B to the identity matrix I
%     It is also possible to set B = A if we are interested in capturing anomalies in X
%
% C - Without prior knowledge, we set C to be the identity matrix.
% 
%Output:
% X - a low-rank matrix
%
% Y - a sparse anomaly matrix
%
% Z -  a noise matrix
%
% W -  an error matrix
%
%
%others:
%
% M, {M k }(k=0,1), N are the Lagrangian multipliers
% |P1 * X1 * Q'1 - R1|_F^2 - capture domain knowledge
%
% Algorithm:
%
% Initialize X,Y,Z,W
% while ~converged 
%   minimize (inexactly, update A and E only once)
%     L(X,X0,X1,Y,Y0,Z,W,M,M0,M1,N,mu) = aerf * |X|_* + beita * |Y|_1 + 1/(2*sigama) * |Z|_F^2
%                  + gama/(2*sigama) * |P1 * X1 * Q'1 - R1|_F^2
%                  + <M , D - A*X0 - B*Y0 - C*Z - W> + <M0 , X0 - X>
%                  + <M1 , X1 - X> + <N , Y0 - Y> + mu/2 * |D - A*X0 - B*Y0 - C*Z - W|_F^2
%                  + mu/2 * |X0 - X|_F^2 + mu/2 * |X1 - X|_F^2 +mu/2 * |Y0 - Y|_F^2;
% 
%   J1 = (X0 + M0/mu + X1 + M1/mu)/(K+1) , tx = (aerf/mu)/(K+1) X = SVSoftThresh(J1,tx);
% 
%   J2 = X - M1/mu , J2 = X - M1/mu , R = P1'*R1*Q1 + (mu*sigama/gama)*J2;
%   [U,S]=eig(P1'*P1);[V,T]=eig(Q1'*Q1) , s = diag(S); t = diag(T) , X1 =U*((U\R*V)./(s*t'+mu*sigama/gama))/V
% 
%   J00 = X - M0/mu , J3 = D1 - B*Y0 - C*Z - W + M/mu , X0 = inv(A'*A+eye(m))*(A'*J3+J00);
%
%   J4 = Y0 + N/mu , ty = beita/mu , Y = SoftThresh(J4,ty)
%    
%   J01 = Y - N/mu , J5 = D1 - A*X0 - C*Z - W + M/mu , Y0 = inv(B'*B+eye(m))*(B'*J5+J01)
%    
%   J6 = D1 - A*X0 - B*Y0 - W + M/mu , Z = inv(eye(m)/(mu*sigama)+C'*C)*(C'*J6)
%   
%   W = E .* (D1 - A*X0 - B*Y0 - C*Z + M/mu)
%    
%   sigama = sita * sigamad
%
%   M = M + mu*(D1 - A*X0 - B*Y0 - C*Z - W) , M0 = M0 + mu*(X0 - X_hat),
%   M1 = M1 + mu*(X1 - X) , N = N + mu*(Y0 - Y);
%
%   mu = mu * rou
%
% end
%
% 
% paper:http://dx.doi.org/10.1145/2639108.2639129.

%

addpath PROPACK;

[m , n] = size(D);
D1 = D;
D2 = D;
D1(isnan(D1))=0;
D2(~isnan(D2))=0;
D2(isnan(D2))=1;
if nargin < 2
    gama = 1 / sqrt(m);
end

if nargin < 3
    tol = 1e-7;
elseif tol == -1
    tol = 1e-7;
end

if nargin < 4
    maxIter = 1000;
elseif maxIter == -1
    maxIter = 1000;
end

if nargin < 5
    E = D2;
elseif E == -1
    E = D2;;
end

if nargin < 6
    %A = eye(m,n);
    A = eye(m);
elseif A == -1
    %A = eye(m,n);
    A = eye(m);
end

if nargin < 7
    %B = eye(m,n);
    B = eye(m);
elseif B == -1
    %B = eye(m,n);
    B = eye(m);
end

if nargin < 8
    %C = eye(m,n);
    C = eye(m);
elseif C == -1
    %C = eye(m,n);
    C = eye(m);
end


d_norm = norm(D1, 'fro');
% initialize
%how to initialize M,Mk,N?
M = D1;
norm_two = lansvd(M, 1, 'L');
norm_inf = norm( M(:), inf) / gama;
dual_norm = max(norm_two, norm_inf);
M = M / dual_norm;

% initialize others
X_hat = zeros( m, n);
Y_hat = zeros( m, n);
Z_hat = zeros( m, n);
W_hat = zeros( m, n);
X0 = X_hat;
X1 = X_hat;
Y0 = Y_hat;
M0 = X0 - X_hat;
M1 = X1 - X_hat;
N = Y0 - Y_hat;
[mx , nx] = size(X_hat);
[my , ny] = size(Y_hat);
sita = 10;
etad = 1 - sum(sum(E))/(m*n);
sigamad = std(reshape(D1,m*n,1));
mu = 1.01; % this one can be tuned
mu_bar = mu * 1e7;
rou = 1.01;          % this one can be tuned
aerf = (sqrt(mx) + sqrt(nx)) * sqrt(etad);
beita = sqrt(2 * log(my * ny));
sigama = sita * sigamad;

K = 1;
%initialize temporal locality
P1 = eye(m);
aa = eye(n-1);
bb = zeros(1,n-1);
cc = [bb' aa];
bb = zeros(1,n);
Q1 = [cc;bb];
R1 = zeros(m,n);
% %initialize spatial locality
% Q1 = eye(m);
% aa = eye(n-1);
% bb = zeros(1,n-1);
% cc = [bb' aa];
% bb = zeros(1,n);
% P1 = [cc;bb];
% R1 = zeros(m,n);
iter = 0;

converged = false;
stopCriterion = 1;

while ~converged       
    iter = iter + 1;
    if mod(iter,100) == 0
        rou = rou * 1.05;
    end
    %update X
    J1 = (X0 + M0/mu + X1 + M1/mu)/(K+1);
    tx = (aerf/mu)/(K+1);
    [Ux Sx Vx] = svd(J1, 'econ');
    X_hat = max(Sx - tx, 0);
    X_hat = X_hat+min(Sx + tx, 0);
    %update Xk
    J2 = X_hat - M1/mu;
    R = P1'*R1*Q1 + (mu*sigama/gama)*J2;
    [U,S]=eig(P1'*P1);[V,T]=eig(Q1'*Q1);
    s = diag(S); t = diag(T);
    X1 = U*((U\R*V)./(s*t'+mu*sigama/gama))/V;
    %update X0
    J00 = X_hat - M0/mu;
    J3 = D1 - B*Y0 - C*Z_hat - W_hat + M/mu;
    X0 = inv(A'*A+eye(m))*(A'*J3+J00);
    %update Y
    J4 = Y0 + N/mu;
    ty = beita/mu;
    [Uy Sy Vy] = svd(J4, 'econ');
    Y_hat = max(Sy - ty, 0);
    Y_hat = Y_hat+min(Sy + ty, 0);
    %update Y0
    J01 = Y_hat - N/mu;
    J5 = D1 - A*X0 - C*Z_hat - W_hat + M/mu;
    Y0 = inv(B'*B+eye(m))*(B'*J5+J01);
    %update Z
    J6 = D1 - A*X0 - B*Y0 - W_hat + M/mu;
    Z_hat = inv(eye(m)/(mu*sigama)+C'*C)*(C'*J6);
    %update W
    W_hat = E .* (D1 - A*X0 - B*Y0 - C*Z_hat + M/mu);
    %update sigamad
    J8 = D1 - A*X0 - B*Y0 - W_hat;
    sigamad = std(J8(E==0));
    sigama = sita * sigamad;
    %update M,Mk,N
    M = M + mu*(D1 - A*X0 - B*Y0 - C*Z_hat - W_hat);
    M0 = M0 + mu*(X0 - X_hat);
    M1 = M1 + mu*(X1 - X_hat);
    N = N + mu*(Y0 - Y_hat);
    %update mu
    mu = mu * rou;
    J = D - A*X_hat - B*Y_hat - C*Z_hat - W_hat;
    %% stop Criterion    
    stopCriterion = norm(J, 'fro') / d_norm;
    if stopCriterion < tol
        converged = true;
    end    
        
    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end
