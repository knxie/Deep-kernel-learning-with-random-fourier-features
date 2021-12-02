load('cifar_10.mat' ,'X_data_train', 'X_data_test');
load('cifar_test.mat','X_data_test');



for i = 1:5
    X =[ X_data_train{i};X_data_train{i}];
    X_test = [ X_data_test{i};X_data_test{i}];
    Y = [ones(5000,1);-1*ones(5000,1)];
    Y_test = [ones(1000,1);-1*ones(1000,1)];
    
    flagtest =1;
    
    save(['cifar_' num2str(i) '.mat']);
end