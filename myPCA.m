
% input dataset and number of PC wanted
% output eigenvectors and corresponding eigenvalues
function [myPCA] = myPCA(x, dim)
cov_x=cov(x(:,1:end-1));
[V_x,lambda_x] = eig(cov_x);
V_x_descend=fliplr(V_x);
[l, idx] = sort(diag(lambda_x)');
l = fliplr(l);
labmda=l(1:dim) % eigenvalues
V=V_x_descend(:,1:dim)  %eigenvectors
end