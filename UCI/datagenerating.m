close all
clear
clc

% ------ AC170-S060 ?4-class? ---
% load AC170-S060-1
% X = MPAnP208pS15Homo;
% load AC170-S060-2
% X = [X; MPAnP208pS15Homo(2:end, :)];
% load AC170-S060-3
% X = [X; MPAnP208pS15Homo(2:end, :)];
% 
% Y = X(:, end);
% X = X(:, 1 : end);
% 
% indx = find(Y > 0);
% Y = Y(indx);
% X = X(indx, :);
% 
% Y(find(Y <= 1)) = -1;
% Y(find(Y > 1)) = 1;
% % 
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end));



% ----- LFSR-32 --------
% load lfsr32_test
% X = x_total;
% Y = y_total;
% Y(find(Y==0)) = -1;
% % 
% [temp, index] = sort(rand( length(Y), 1));
% X = X( index( 1 : Training_num ), : );
% Y = Y( index( 1 : Training_num));
% 
% % X = X( index( end + 1 - Training_num : end), : );
% % Y = Y( index( end + 1 - Training_num : end));
% 
% load lfsr32_test
% X_test = x_training;
% Y_test = y_training;
% Y_test(find(Y_test==0)) = -1;

%------- LFSR -----------
% load lfsr8
% X = x(1:99,4:8);
% Y = y;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X = X( index( end + 1 - Training_num : end), : );
% Y = Y( index( end + 1 - Training_num : end));
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : Training_num ), : );
% Y_test = Y( index( 1 : Training_num));
% 
% X = X( index( end + 1 - Training_num : end), : );
% Y = Y( index( end + 1 - Training_num : end));

%---- fertility ------
% load fertility.txt
% X = fertility( :, 1 : end - 1 );
% Y = fertility( :, end );
% flagtest = 0;
% save fertility X Y flagtest
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end));  

% --- DB words -------
% load dbworld_bodies
% load dbworld_bodies_stemmed
% load dbworld_subjects
% load dbworld_subjects_stemmed
% % 
% X = inputs;
% Y = labels;
% Y(find(Y==0)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : Training_num ), : );
% Y_test = Y( index( 1 : Training_num));
% 
% X = X( index( end + 1 - Training_num : end), : );
% Y = Y( index( end + 1 - Training_num : end)); 

% --- Planing Relax -------
% load planning.txt
% X = planning( :, 1 : end - 1 );
% Y = planning( :, end );
% Y(find(Y==1)) = 1;
% Y(find(Y==2)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 

% --- sonar ----------
% load sonar.txt
% X = sonar( :, 1 : end - 1 );
% Y = sonar( :, end );
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 

% ---- Liver -------
% load liver.mat
% X = liver( :, 1 : end - 1 );
% Y = liver( :, end );
% Y(find(Y==1)) = 1;
% Y(find(Y==2)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 

% ---- Phishing -------
% load phishing.txt
% X = phishing( :, 1 : end - 1 );
% Y = phishing( :, end );
% Y(find(Y==1)) = 1;
% Y(find(Y==2)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 

% ------- diabetic -----
% load diabetic.txt
% X = diabetic( :, 1 : end - 1 );
% Y = diabetic( :, end );
% Y(find(Y==1)) = 1;
% Y(find(Y==0)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 

% ------ EEG ---------
% load EEG.txt
% X = EEG( :, 1 : end - 1 );
% Y = EEG( :, end );
% Y(find(Y==1)) = 1;
% Y(find(Y==0)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 

% ----- climate ------
% load climate.txt
% X = climate( :, 1 : end - 1 );
% Y = climate( :, end );
% Y(find(Y==1)) = 1;
% Y(find(Y==0)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 


% ------ QSAR ---------
% load QSAR.txt
% X = QSAR( :, 1 : end - 1 );
% Y = QSAR( :, end );
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 


% ------- Wilt -----
% load wilt_train.txt
% X = wilt_train( :, 2 : end );
% Y = wilt_train( :, 1 );
% 
% load wilt_test.txt
% X_test = wilt_test( :, 2 : end );
% Y_test = wilt_test( :, 1);

% --- steel ------
% load steel.txt
% X = steel( :, 1 : end - 1 );
% Y = steel( :, end );
% Y(find(Y==1)) = 1;
% Y(find(Y==0)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 

% --- mlprove ------
% load mlprove_train.txt
% X = mlprove_train( :, 1 : end - 6 );
% Y = mlprove_train( :, end );
% 
% load mlprove_test.txt
% X_test = mlprove_test( :, 1 : end - 6 );
% Y_test = mlprove_test( :, end );

% --------- Medelon ---------
% load madelon_train.txt
% X = madelon_train;
% load madelon_train_Y.txt
% Y = madelon_train_Y;
% load madelon_valid.txt
% X_test = madelon_valid;
% load madelon_test_Y.txt
% Y_test = madelon_test_Y;

% ----------- UJIndoor ---------
% load UJindoor.txt
% X = UJindoor( :, 1 : end - 9 );
% Y = UJindoor( :, end - 3);
% Y(find(Y==1)) = 1;
% Y(find(Y==2)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 

%------- IJCNN 1 --------
% load ijcnn_tr.mat
% load ijcnn_test.mat
% [temp, index] = sort(rand( length(Y), 1));
% X = X( index( end + 1 - Training_num : end), : );
% Y = Y( index( end + 1 - Training_num : end));
% 
% index_non_zero = find(Y ~= 0);
% X = X(index_non_zero, :);
% Y = Y(index_non_zero);
% 
% index_non_zero = find(Y_test ~= 0);
% X_test = X_test(index_non_zero, :);
% Y_test = Y_test(index_non_zero);
% 
% 
% X = full(X);
% Y = full(Y);
% X_test = full(X_test);
% Y_test = full(Y_test);


%------- real-sim --------
% load real_sim.mat
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : Training_num ), : );
% Y_test = Y( index( 1 : Training_num));
% 
% X = X( index( end + 1 - Training_num : end), : );
% Y = Y( index( end + 1 - Training_num : end));



% --------- SVM Guide 1------------
% load guide1.txt
% X = guide1( :, 2 : end );
% Y = guide1( :, 1 );
% Y(find(Y==0)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X = X( index( end + 1 - Training_num : end), : );
% Y = Y( index( end + 1 - Training_num : end)); 
% 
% load guide1_t.txt
% X_test = guide1_t( :, 2 : end);
% Y_test = guide1_t( :, 1);
% Y_test(find(Y_test==0)) = -1;

% --------SVM Guide 3-------------
% load guide3.txt
% X = guide3( :, 3 : 2 : end - 1 );
% Y = guide3( :, 1 );
% 
% % [temp, index] = sort(rand( length(Y), 1));
% % X = X( index( end - Training_num : end), : );
% % Y = Y( index( end - Training_num : end)); 
% 
% load guide3_t.txt
% X_test = guide3_t( :, 3 : 2 : end - 1);
% Y_test = guide3_t( :, 1);

% -------------------------
% splice 
% load splice.txt
% X = splice( :, 3 : 2 : end);
% Y = splice( :, 1);
% 
% [temp, index] = sort(rand( length(Y), 1));
% X = X( index( end + 1 - Training_num : end), : );
% Y = Y( index( end + 1 - Training_num : end)); 
%  
% load splice_t.txt
% X_test = splice_t( :, 3 : 2 : end);
% Y_test = splice_t( :, 1);

% -------------------
% cod_rna 
% load cod_rna.txt
% X = cod_rna( :, 2 : end);
% Y = cod_rna( :, 1);
% 
% % [temp, index] = sort(rand( length(Y), 1));
% % X = X( index( end - Training_num : end), : );
% % Y = Y( index( end - Training_num : end)); 
%  
% load cod_rna_t.txt
% X_test = cod_rna_t( :, 2 : end);
% Y_test = cod_rna_t( :, 1);



% load australian.txt
% X = australian( :, 2 : end );
% Y = australian( :, 1 );
%  
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 



% load pima_indians_diabetes.txt
% X = pima_indians_diabetes( :, 1 : end - 1 );
% Y = pima_indians_diabetes( :, end );
% Y(find(Y==0)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 

% load breast_cancer_wisconsin.txt
% X = breast_cancer_wisconsin( :, 1 : end - 1 );
% Y = breast_cancer_wisconsin( :, end );
% Y(find(Y==2)) = 1;
% Y(find(Y==4)) = -1;
% flagtest = 0;
% save breast_cancer_wisconsin X Y flagtest
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end));  


% monks_1 
% load monks_1_train.txt
% X = monks_1_train( :, 2 : end - 1);
% Y = monks_1_train( :, 1);
% Y(find(Y==0)) = -1;
% 
% Training_num = min(Training_num, length(Y));
% [temp, index] = sort(rand( length(Y), 1));
% X = X( index( end - Training_num + 1: end), : );
% Y = Y( index( end - Training_num + 1: end));
% %  
% load monks_1_test.txt
% X_test = monks_1_test( :, 2 : end - 1);
% Y_test = monks_1_test( :, 1);
% Y_test(find(Y_test==0)) = -1;
%------------------------------------------

%monks_2 
% load monks_2_train.txt
% X = monks_2_train( :, 2 : end - 1);
% Y = monks_2_train( :, 1);
% Y(find(Y==0)) = -1;
% 
% Training_num = min(Training_num, length(Y));
% [temp, index] = sort(rand( length(Y), 1));
% X = X( index( end - Training_num + 1: end), : );
% Y = Y( index( end - Training_num + 1: end));
% 
% load monks_2_test.txt
% X_test = monks_2_test( :, 2 : end - 1);
% Y_test = monks_2_test( :, 1);
% Y_test(find(Y_test==0)) = -1;

% monks_3 
% load monks_3_train.txt
% X = monks_3_train( :, 2 : end - 1);
% Y = monks_3_train( :, 1);
% Y(find(Y==0)) = -1;
% 
% Y(1) = -Y(1);
% 
% load monks_3_test.txt
% X_test = monks_3_test( :, 2 : end - 1);
% Y_test = monks_3_test( :, 1);
% Y_test(find(Y_test==0)) = -1;



%magic
% load magic04.txt
% X = magic04( :, 1 : end - 1 );
% Y = magic04( :, end );
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end));

% SPECT
% load SPECT_train.txt
% X = SPECT_train( :, 2 : end - 1);
% Y = SPECT_train( :, 1);
% Y(find(Y==0)) = -1;
% % 
% load SPECT_test.txt
% X_test = SPECT_test( :, 2 : end - 1);
% Y_test = SPECT_test( :, 1);
% Y_test(find(Y_test==0)) = -1;



%transfusion
% load transfusion.txt
% X = transfusion( : , 1 : end - 1 );
% Y = transfusion( :, end );
% Y(find(Y==0)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 

% haberman
% load haberman.txt
% X = haberman( : , 1 : end - 1 );
% Y = haberman( :, end );
% Y(find(Y==2)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 

% % ionosphere
% load ionosphere.txt
% X = [ionosphere( : , 1), ionosphere( :, 3 : end - 1 )];
% Y = ionosphere( :, end );
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 
% 
% X_test = X;
% Y_test = Y;

% load spambase.txt
% X = [spambase( : , 1 : end - 1 )];
% Y = spambase( :, end );
% Y(find(Y==0)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 



% load parkinsons.txt
% X = [parkinsons( : , 1 : 16 ), parkinsons( : , 17 : end )];
% Y = parkinsons( :, 17 );
% Y(find(Y==0)) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : 97), : );
% Y_test = Y( index( 1 : 97));
% X = X( index( 98 : end), : );
% Y = Y( index( 98 : end)); 


% statlog
% load heart.txt
% X = heart( : , 1 : end - 1 );
% Y = heart( :, end );
% Y(find(Y==2)) = -1;
% % 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - Training_num + 1), : );
% Y_test = Y( index( 1 : end - Training_num + 1));
% X = X( index( end - Training_num : end), : );
% Y = Y( index( end - Training_num : end)); 




% -------------- 2 D ------------------
% fourclasses
% load fourclasses.txt
% X = fourclasses( :, 2 : end );
% Y = fourclasses( :, 1 );
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : end - 100), : );
% Y_test = Y( index( 1 : end - 100));
% X = X( index( end - 99 : end), : );
% Y = Y( index( end - 99 : end));


% load fisheriris
% X = [meas(:,1), meas(:,2)];
% Y = zeros( size(X,1), 1 );
% Temp = nominal(ismember(species,'setosa'));
% Y(find(Temp == 'true')) = 1;
% Y(find(Temp == 'false')) = -1;
% 
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : 75), : );
% Y_test = Y( index( 1 : 75));
% X = X( index( 76 : end), : );
% Y = Y( index( 76 : end)); 


% two-moon
% nb = 500;
% sig = 0.2;
% leng = 1;
% for t=1:nb, 
%   yin(t,:) = [2.*sin(t/nb*pi*leng) 2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
%   yang(t,:) = [-2.*sin(t/nb*pi*leng) .45-2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
%   samplesyin(t,:)  = [yin(t,1)+yin(t,3).*randn   yin(t,2)+yin(t,3).*randn];
%   samplesyang(t,:) = [yang(t,1)+yang(t,3).*randn   yang(t,2)+yang(t,3).*randn];
% end
% 
% %X = [samplesyin; samplesyang( 1 : 200, :)];
% X = [samplesyin; samplesyang];
% Y = ones( size(X,1), 1);
% Y( 1 : size( samplesyin, 1 ) ) = -1;
% 
% [temp, index] = sort(rand( length(Y), 1));
% X_test = X( index( 1 : 200), : );
% Y_test = Y( index( 1 : 200));
% X = X( index( 201 : end), : );
% Y = Y( index( 201 : end)); 
% 
% 
% 
% X = [X; X(1:30, :)];
% Y = [Y; Y(1:30)];

% x=0:0.05:2*pi;
% c=cos(x);
% s=sin(x);
% ss=s(find(s.*(x>=0&x<=3)));
% cc=c(find(c.*(x>=1.5&x<=4.5)));
% xc=x(find(x.*(x>=1.5&x<=4.5)));
% xs=x(find(x.*(x>=0&x<=3)));
% s1=randn(size(ss))./10+ss;
% c1=randn(size(cc))./10+cc;
% plot(xc,c1,'ro');
% hold on
% plot(xs,s1,'bd');
% 
% X1 = [xc', c1'];
% X2 = [xs', s1'];
% X = [X1; X2];
% Y = [ones(length(X1), 1); -ones(length(X2), 1)];
% 
% X_test = X;
% Y_test = Y;

% ---- Horse Shoe Shaped data ---------
% density = 100;
% X = [];
% temp = rand( density, 2 );
% X_added = [ temp(:,1)*0.15, temp(:,2)*0.85 + 0.15, ]; % 0.15~1; 0~0.15
% X = [X; X_added];
% temp = rand( density, 2 );
% X_added = [ temp(:,1)*0.7 + 0.15, temp(:,2)*0.15 + 0.85]; %0.15~0.85; 0.85~1
% X = [X; X_added];
% temp = rand( density, 2 );
% X_added = [ temp(:,1)*0.15 + 0.85, temp(:,2)*0.85 + 0.15];% 0.85~1; 0.15~1
% X = [X; X_added];
% temp = rand( density, 2 );
% X_added = [ temp(:,1)*0.2 + 0.4, temp(:,2)*0.85 + 0.15];% 0.4~0.6; 0.15~0.85
% X = [X; X_added];
% 
% X1 = X;
% Y1 = ones( size(X1,1),1);
% 
% figure
% hold on
% plot(X(:,1),X(:,2),'ro');
% axis([0 1 0 1])
% 
% X = [];
% temp = rand( density, 2 );
% X_added = [ temp(:,1)*0.15 + 0.2, temp(:,2)*0.8, ]; % 0.2~0.35; 0~0.8
% X = [X; X_added];
% temp = rand( density, 2 );
% X_added = [ temp(:,1)*0.15 + 0.65, temp(:,2)*0.8 ]; %0.65~0.8; 0~0.8
% X = [X; X_added];
% temp = rand( density, 2 );
% X_added = [ temp(:,1)*0.3 + 0.35, temp(:,2)*0.1];% 0.35~0.65; 0~0.1
% X = [X; X_added];
% 
% 
% X2 = X;
% Y2 = -ones( size(X2,1),1);
% plot(X(:,1),X(:,2),'bd');
% 
% X = [X1; X2];
% Y = [Y1; Y2];
% 
% X_test = X;
% Y_test = Y;


% ---- two squares, convexly seperable ----------
% N_s = 1000;
% X = rand( N_s, 2);
% Y = zeros( N_s, 1 );
% for ii = 1 : N_s
%     d = max( abs( X( ii, 1 ) - 0.5 ), abs( X( ii, 2 ) - 0.5 ) );
%     if d < 0.18
%         Y(ii) = 1;
%     else if d > 0.22
%             Y(ii) = -1;
%         else
%             ;%Y(ii) = sign( rand - 0.5 );
%         end
%     end
%     if rand < 0.15
%         Y(ii) = sign( rand - 0.5 );
%     end
% end
% index = find( Y ~= 0 );
% X = X( index, :);
% Y = Y( index );
% 
% X_test = X( 1 : 500, :) ;
% Y_test = Y( 1 : 500);
% X = X( 501 : end, :);
% Y = Y( 501 : end );



% ---- two circles, convexly non-seperable ----------
% N_s = 500;
% X = rand( N_s, 2);
% Y = zeros( N_s, 1 );
% for ii = 1 : N_s
%     d = min( sum( (X(ii,:) - [0.2, 0.2]).^2 ), sum( (X(ii,:) - [0.8, 0.8]).^2 )  );
%     if d < 0.03
%         Y(ii) = 1;
%     else if d > 0.04
%             Y(ii) = -1;
%         else
%             ;%Y(ii) = sign( rand - 0.5 );
%         end
%     end
% end
% index = find( Y ~= 0 );
% X = X(index, :);
% Y = Y( index );
% 
% X_test = X;
% Y_test = Y;

% ------ sqaure, half circle ------------
% N_s = 1000;
% X = [];
% Y = [];
% figure
% hold on
% axis( [0 1 0 1]);
% for ii = 1 : N_s
%     temp = rand( 1, 2 );
%     y = 0.1*( sin( 10 * log(1 - 1/50 + temp(1)*20)) + 1) + 0.5;
%     if y - temp(2) >  0.025
%         X = [X; temp];
%         Y = [Y; 1];
%         plot( temp(1), temp(2), '*');
%     end
%     if y - temp(2) < -0.025
%         X = [X; temp];
%         Y = [Y; -1];
%         plot(  temp(1), temp(2), 'o');
%     end
% end
% 
% X_test = X( 1 : 500, :);
% Y_test = Y( 1 : 500, :);
% 
% X = X(501 : end, :);
% Y = Y(501 : end, :);

% ------ Toy Example ------------
% X = 2.*rand(200,2)-1;
% %Y = sign(sin(X(:,1))+X(:,2));
% f_temp = X(:,2) - X(:,1).^3;
% Y = sign(f_temp);
% 
% boundary_index = find(abs(f_temp) < 0.05 );
% no_boundary_index = find(abs(f_temp) > 0.05 );

%Y(boundary_index) = - Y(boundary_index) ;
%Y(boundary_index) = sign( rand(length(boundary_index),1) - 0.5 );

% X_noise = X;
% Y_noise = Y;
% 
% X = X(no_boundary_index, :);
% % Y = Y(no_boundary_index);
% 
% X_test = X( 1 : end, :);
% Y_test = Y( 1 : end, :);


% ---- Clowns ---------
% [X,Y, X_test, Y_test]=datasets('Clowns',500,5000,2);
% 

% ------ Checkers -------
% [X,Y, X_test, Y_test]=datasets('Checkers',500,6000,2);
 
% ------- Gaussian -------
% [X,Y, X_test, Y_test]=datasets('gaussian', 500, 5000);

% ------- CosExp -------

% [X,Y,X_test,Y_test]=datasets('CosExp',500, 5000);

% ------- mixture --------
% [X,Y,X_test,Y_test]=datasets('mixture',50, 5000);


% temp = rand(size(X_test, 1),1);
% [~,index] = sort(temp);
% index = index(1 : Training_num);
% X = X_test( index, : );
% Y = Y_test( index, :);

% ----- SC ----------
% mu = [0 0]; Sigma = [1 .5; .5 1];
% X = mvnrnd(mu, Sigma, 500);
% plot(X(:,1),X(:,2),'.');
% hold on
% Y = -ones( 500, 1 );
% for ii = 1 : 500
%     if rand < 1/(1 + exp(X(ii,1) - X(ii,2) - 0.5))
%         Y(ii) = 1;
%     end
% end
% xlabel('x1')
% ylabel('x2')
% 
% 
% 
% hold on
% for ii = 1 : size( X, 1)
%     if Y(ii) == 1
%         plot(X(ii,1), X(ii,2), 'g*')
%     else
%         plot(X(ii,1), X(ii,2), 'r+')
%     end
% end
% 
% X_test = X;
% Y_test = Y;

% --- aLS example -----------
% mu = [-0.5 3]; Sigma = [0.2 0; 0 3];
% X = mvnrnd(mu, Sigma, ceil(Training_num/2*(1-xi)));
% mu = [0 0]; Sigma = [1 -0.8; -0.8 1];
% X_n = mvnrnd(mu, Sigma, ceil(Training_num/2*xi));
% 
% % plot(X(:,1),X(:,2),'.');
% % hold on
% Y = -ones( ceil(Training_num/2*(1-xi)), 1 );
% 
% X = [X; X_n];
% Y = [Y;-ones( ceil(Training_num/2*xi), 1 )];
% 
% mu = [0.5 -3]; Sigma = [0.2 0; 0 3];
% X2 = mvnrnd(mu, Sigma, ceil(Training_num/2*(1-xi)));
% mu = [0 0]; Sigma = [1 -0.8; -0.8 1];
% X_n = mvnrnd(mu, Sigma, ceil(Training_num/2*xi));
% X = [X; X2];
% % plot(X(:,1),X(:,2),'.');
% % hold on
% Y = [Y; ones( ceil(Training_num/2*(1-xi)), 1 )];
% 
% X = [X; X_n];
% Y = [Y;ones( ceil(Training_num/2*xi), 1 )];
% 
% % 
% % 
% hold on
% for ii = 1 : size( X, 1)
%     if Y(ii) == 1
%         plot(X(ii,1), X(ii,2), 'g*')
%     else
%         plot(X(ii,1), X(ii,2), 'r+')
%     end
% end
% % 
% % X_test = X;
% % Y_test = Y;


% ------- aLS example 2 ----------------
% mu = [-0.5 3]; Sigma = 1*[0.2 0; 0 3];
% X = mvnrnd(mu, Sigma, Training_num/2);
% Y = -ones( Training_num/2, 1 );
%
% mu = [0.5 -3]; Sigma = 1*[0.2 0; 0 3];
% X2 = mvnrnd(mu, Sigma, Training_num/2);
% X = [X; X2];
% Y = [Y;ones( Training_num/2, 1 )];
%
% mu = [-0.5 3]; Sigma = 1*[0.2 0; 0 3];
% X_test = mvnrnd(mu, Sigma, 1000);
% Y_test = -ones( 1000, 1 );
%
% mu = [0.5 -3]; Sigma = 1*[0.2 0; 0 3];
% X2 = mvnrnd(mu, Sigma, 1000);
% X_test = [X_test; X2];
% Y_test = [Y_test;ones( 1000, 1 )];
%  
% % 
% % 
% figure
% axis([-2 2 -6 8])
% hold on
% for ii = 1 : size( X, 1)
%     if Y(ii) == 1
%         plot(X(ii,1), X(ii,2), 'g*')
%     else
%         plot(X(ii,1), X(ii,2), 'r+')
%     end
% end
% 
% % ---------------------

% ------- aLS example 2 ----------------
% mu = [-0.5 3]; Sigma = 1*[0.2 0; 0 3];
% X = mvnrnd(mu, Sigma, Training_num/2);
% Y = -ones( Training_num/2, 1 );
% 
% mu = [0.5 -3]; Sigma = 1*[0.2 0; 0 3];
% X2 = mvnrnd(mu, Sigma, Training_num/2);
% X = [X; X2];
% Y = [Y;ones( Training_num/2, 1 )];
% 
% 
% sigma = rand(Training_num, 1) * 0.7 + 0.1;
% [~, index_temp] = sort(rand(Training_num, 1));
% sigma( index_temp( 1 : ceil(Training_num * 0.1))) = rand(ceil(Training_num * 0.1), 1) * 1.5 + 0.5;
% X_noise = X;
% for ii = 1 : Training_num
%     X_noise( ii, :) = X( ii, : ) + mvnrnd([0 0], sigma(ii) * eye(2), 1);
% end
% 
% % X = X_noise;
% 
% 
% mu = [-0.5 3]; Sigma = 1*[0.2 0; 0 3];
% X_test = mvnrnd(mu, Sigma, 2500);
% Y_test = -ones( 2500, 1 );
% 
% mu = [0.5 -3]; Sigma = 1*[0.2 0; 0 3];
% X2 = mvnrnd(mu, Sigma, 2500);
% X_test = [X_test; X2];
% Y_test = [Y_test;ones( 2500, 1 )];


%------ p.d.f plot -------------
% xx = 0 : 0.001 : 1;
% [xx1, xx2] = meshgrid(xx);
% xx1 = (xx1 - 0.5) * 4;
% xx2 = (xx2 - 0.5) * 16;
% yy1 = mvnpdf([xx1(:), xx2(:)], [-0.5 3], Sigma);
% yy2 = mvnpdf([xx1(:), xx2(:)], [0.5 -3], Sigma);
% yy_n = mvnpdf([xx1(:), xx2(:)], [0 0], [1 -0.8; -0.8 1]);
% yy1_mesh = reshape(yy1, length(xx), length(xx)); 
% yy2_mesh = reshape(yy2, length(xx), length(xx));
% yy_n_mesh = reshape(yy_n, length(xx), length(xx));
% 
% yy4 = 1/10/10 * ones(size(xx1));
% yy4(find(xx1 > xx2)) = 0;
% yy5 = 1/10/10 * ones(size(xx1));
% yy5(find(xx1 < xx2)) = 0;

% figure
% contour(xx1, xx2, yy1_mesh, 8)
% hold on
% contour(xx1, xx2, -yy2_mesh, 8)
% axis equal
% axis([-2 2 -7 7])
% 
% figure
% contour(xx1, xx2, (1-xi)*yy1_mesh + xi*yy_n_mesh, 8)
% hold on
% contour(xx1, xx2, -(1-xi)*yy2_mesh - xi*yy_n_mesh, 8)
% axis equal
% axis([-2 2 -7 7])

% figure
% mesh(xx1, xx2, yy4)
% hold on
% mesh(xx1, xx2, -yy5)
% axis equal
% axis([-2 2 -7 7])


% %
% %
% figure
% axis equal
% axis([-2 2 -8 8])
% 
% hold on
% for ii = 1 : size( X, 1)
%     if Y(ii) == 1
%         plot(X(ii,1), X(ii,2), 'g*')
%     else
%         plot(X(ii,1), X(ii,2), 'r+')
%     end
% end
% line([-2; 2], [-5; 5])
% 
% figure
% axis([-2 2 -6 8])
% hold on
% for ii = 1 : size( X, 1)
%     if Y(ii) == 1
%         plot(X_noise(ii,1), X_noise(ii,2), 'g*')
%     else
%         plot(X_noise(ii,1), X_noise(ii,2), 'r+')
%     end
% end
% 

% -------------------------------------------------------
% X = (rand(Training_num, 2)  - 0.5)*10;
% Y = X(:,1) - X(:,2);
% Y( find(Y > 0) ) = 1;
% Y( find(Y < 0 )) = -1;
% sigma = rand(Training_num, 1) * 0.7 + 0.1;
% [~, index_temp] = sort(rand(Training_num, 1));
% sigma( index_temp( 1 : ceil(Training_num * 0.1))) = rand(ceil(Training_num * 0.1), 1) * 1.5 + 0.5;
% X_noise = X;
% for ii = 1 : Training_num
%     X_noise( ii, :) = X( ii, : ) + mvnrnd([0 0], sigma(ii) * eye(2), 1);
% end
% 
% X = X_noise;
% 
% X_test = (rand(10000, 2)  - 0.5)*10;
% Y_test = X_test(:,1) - X_test(:,2);
% Y_test( find(Y_test > 0) ) = 1;
% Y_test( find(Y_test < 0 )) = -1;
% 

%---------------


% X_orignal = X;
% Y_orignal = Y;

% X_test = [X_test; X];
% Y_test = [Y_test; Y];
% 

