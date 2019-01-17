function[p1,p2,pc1,pc2] =Bayes_learning(training_data, validation_data)
% calculate p1 and p2 for Bernoulli distribution 
p1=zeros(size(training_data,2)-1,1);
p2=zeros(size(training_data,2)-1,1);
for j= 1: size(training_data,2)-1
    p1(j)=p1(j)+sum( (training_data(:,j)==0)& (training_data(:,end)==1) )/sum(training_data(:,end)==1);
end
for j= 1: size(training_data,2)-1
    p2(j)=p2(j)+sum( (training_data(:,j)==0)& (training_data(:,end)==2) )/sum(training_data(:,end)==2);
end
% calculate error for each sigma
% initialize error rate matrix and prior
valid_err=zeros(11,1);
Prior=zeros(11,1);
for sigma=-5:5
% calculate the prior probability using sigma+6 to indicate which sigma
% gives the lowest error rate since it will automatically give the index
% of sigma we're using
    Prior(sigma+6)=Prior(sigma+6)+1/(1+exp(-sigma));
%calculate the posterior probability of valid set 
%classes
    P_c_x=ones(size(validation_data, 1),2);

    for n= 1:size(validation_data,1)
        for j=1:(size(validation_data,2)-1)
            P_c_x(n,1)=P_c_x(n,1)* p1(j).^(1-validation_data(n,j))* (1-p1(j)).^(validation_data(n,j));
        end
    end
    P_c_x(:,1)=P_c_x(:,1)*Prior(sigma+6);

    for n= 1:size(validation_data,1)
        for j=1:(size(validation_data,2)-1)
            P_c_x(n,2)=P_c_x(n,2)* p2(j).^(1-validation_data(n,j))* (1-p2(j)).^(validation_data(n,j));
        end
    end
    P_c_x(:,2)=P_c_x(:,2).*(1-Prior(sigma+6));

%compare c1 and c2 column to see the exact class 
    for n=1:size(P_c_x,1)
        if P_c_x(n,1)>P_c_x(n,2)
            P_c_x(n,3)=1;
        else
            P_c_x(n,3)=2;
        end
    end
    
    ss=sum(P_c_x(:,3)~=validation_data(:,end));
    
    valid_err(sigma+6)= valid_err(sigma+6)+ss/size(validation_data,1);
    
end
%find the index of minimium error rate, and corresonding
%prior in class 1 and 2
[min_error,index]=min(valid_err)
pc1=Prior(index)
pc2=1-pc1
valid_err_print=sprintf('%0.5e;',valid_err)
end

