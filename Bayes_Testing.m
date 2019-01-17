function [test_err_print] = Bayes_Testing(test_data,pc1,pc2)
%calculate the posterior probability , 2 classes
P_c_x=ones(size(test_data, 1),2);
% calculate p1 and p2 for Bernoulli densities from test dataset
p1=zeros(size(test_data,2)-1,1);
p2=zeros(size(test_data,2)-1,1);
for j= 1: size(test_data,2)-1
    p1(j)=p1(j)+sum((test_data(:,j)==0)& (test_data(:,end)==1))/sum(test_data(:,end)==1);
end
for j= 1: size(test_data,2)-1
    p2(j)=p2(j)+sum((test_data(:,j)==0)& (test_data(:,end)==2))/sum(test_data(:,end)==2);
end
% calculate posterior probability
for n= 1:size(test_data,1)
    for j=1:(size(test_data,2)-1)
        P_c_x(n,1)=P_c_x(n,1)* p1(j).^(1-test_data(n,j))* (1-p1(j)).^(test_data(n,j));
    end
end
P_c_x(:,1)=P_c_x(:,1)*pc1;

for n= 1:size(test_data,1)
    for j=1:(size(test_data,2)-1)
        P_c_x(n,2)=P_c_x(n,2)* p2(j).^(1-test_data(n,j))* (1-p2(j)).^(test_data(n,j));
    end
end
P_c_x(:,2)=P_c_x(:,2).*pc2;
%compare c1 and c2 column to see the exact class
for n=1:size(P_c_x,1)
    if P_c_x(n,1)>P_c_x(n,2);
        P_c_x(n,3)=1;
    else
        P_c_x(n,3)=2;
    end
 s=sum(P_c_x(:,3)~=test_data(:,end));
end
%calculate the error rate and print
test_err=s/size(test_data,1);
test_err_print=sprintf('%0.5e',test_err)
end

