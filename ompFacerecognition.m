%OMP algorithm for occluded face recognition
%%自定义阶段
%降采样行数和列数
resampled_row = 60;
resampled_col = 40;

%无遮挡的字典的人数
sampleclassnum = 50;  %取值范围[1,100]

%从测试集中选取图像进行测试
test_class = 2;       %选类，取值范围[1,sampleclassnum]
test_seq = 8;        %从该类中选取测试图像，取值范围[8,13]∪[21,26]

%%
%初始化字典
A = [];
% 将无遮挡图像作为训练集
Files = dir(fullfile('.\\AR','*.pgm'));
LengthFiles = length(Files);
Imgorigin = [];
Imgshade = [];
% 
% %展示选取类的所有26张图
% for i = (test_class-1)*26+1:test_class*26
%    Img0 = imread(strcat('.\AR\',Files(i).name));
% if(mod(i,13) >=1 && mod(i,13) <=7)
%    Imgorigin = [Imgorigin,Img0];
% else
%    Imgshade = [Imgshade,Img0];
% end
% end
% figure;
% subplot(211);
% imshow(Imgorigin);
% subplot(212);
% imshow(Imgshade);

%每一类样本的序号1-7，14-20可以当作训练样本加入字典
tic;
for i = 1:sampleclassnum*26
Img0 = imread(strcat('.\AR\',Files(i).name));
if(mod(i,13) >=1 && mod(i,13) <=7)
    Img = imresize(Img0,[resampled_row,resampled_col]);
V = Img(:);
    A =[A,V];
end
end
read_file_time = toc;
fprintf('构建无遮挡字典时间为%.4f秒\n',read_file_time);

A = im2double(A);
[m1,n1] = size(A);

%归一化
A = A./repmat(sqrt(sum(A.^2,1)),size(A,1),1);


%构造被遮挡图像字典
I = im2double(eye(m1));
B = [A,I]; 

if(test_class <= sampleclassnum)
    testImg0 = im2double(imread(strcat('.\AR\',Files(26*(test_class - 1) + test_seq).name)));
    testImg = imresize(testImg0,[resampled_row,resampled_col]);
    [m2,n2] = size(testImg);
    y = testImg(:);
else
    dis('error:inadequate choice of sample classes');
end 

% %ompbox
% ompboxstart = tic;
% EPSILON = 1;
% G = B'* B ;
% gamma = omp2(B'*y,sum(y.*y),G,EPSILON);
% ompbox_time = toc(ompboxstart);
% fprintf('ompbox运算时间共为%.2f秒\n',ompbox_time);

%myOmp
ompstart = tic;
gamma = myOmp(y,B);
omp_time = toc(ompstart);
fprintf('myomp运算时间共为%.2f秒\n',omp_time);

%截取误差向量用于恢复成预测图像
e = gamma(n1 + 1:n1 + m1);
res = y - e;

%量化预测图像和无遮蔽图像的误差
Ediff = repmat(res,14,1) - A(((test_class-1)*14 + 1):test_class*14);
Enor = sqrt(sum(Ediff.^2,1));
Emean = mean(Enor);

%恢复图像
recons = reshape(res,[m2,n2]);
% recons = imresize(recons, 10, 'bicubic');

figure;
subplot(121);
imshow(testImg);title('Test Image');
subplot(122);
imshow(recons);title('Predicted Image');
xlabel(sprintf('(||x_{pred} - x_{origin}||_2)_{avg} = %.2f',Emean));
