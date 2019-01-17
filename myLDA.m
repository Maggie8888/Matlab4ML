function[myLDA]=myLDA(x)
%initialize cov_loop as a uninversible matrix to start the loop
dim=size(x,2)-1;
S_w=zeros(1);
cov_x=cov(x(:,1:end-1));
[V_x,lambda_x] = eig(cov_x);
V_x_descend=fliplr(V_x);
global l vector_LDA_descend vector_LDA 
while rank(S_w)<size(S_w,1)  
   project_loop=x(:,1:end-1)*V_x_descend(:,1:(dim-1)); 
   S_w=zeros(dim-1);
   for c=0:9
       S_w=S_w+cov(project_loop(x(:,end)==c,:));    
   end
   dim=dim-1;
end
% Calculate the between_class scatter
global_mean=mean(project_loop);
S_b=zeros(size(global_mean,2));
for c=0:9
    class_mean=mean(project_loop(x(:,end)==c,:));
    S_b=S_b+sum(x(:,end)==c)*transpose(class_mean-global_mean)*(class_mean-global_mean);
end
%compute a projection using LDA that is largest(last) eigenvectors of 
%inv(S_w)*S_b
[vector_LDA,lambda_LDA]=eig((inv(S_w))*(S_b));
vector_LDA_descend=fliplr(vector_LDA);
[l, idx] = sort(diag(lambda_LDA)');
l = fliplr(l);
end
