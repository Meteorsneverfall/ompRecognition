%OMP algorithm for occluded face recognition
%%�Զ���׶�
%����������������
resampled_row = 60;
resampled_col = 40;

%���ڵ����ֵ������
sampleclassnum = 50;  %ȡֵ��Χ[1,100]

%�Ӳ��Լ���ѡȡͼ����в���
test_class = 2;       %ѡ�࣬ȡֵ��Χ[1,sampleclassnum]
test_seq = 8;        %�Ӹ�����ѡȡ����ͼ��ȡֵ��Χ[8,13]��[21,26]

%%
%��ʼ���ֵ�
A = [];
% �����ڵ�ͼ����Ϊѵ����
Files = dir(fullfile('.\\AR','*.pgm'));
LengthFiles = length(Files);
Imgorigin = [];
Imgshade = [];
% 
% %չʾѡȡ�������26��ͼ
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

%ÿһ�����������1-7��14-20���Ե���ѵ�����������ֵ�
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
fprintf('�������ڵ��ֵ�ʱ��Ϊ%.4f��\n',read_file_time);

A = im2double(A);
[m1,n1] = size(A);

%��һ��
A = A./repmat(sqrt(sum(A.^2,1)),size(A,1),1);


%���챻�ڵ�ͼ���ֵ�
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
% fprintf('ompbox����ʱ�乲Ϊ%.2f��\n',ompbox_time);

%myOmp
ompstart = tic;
gamma = myOmp(y,B);
omp_time = toc(ompstart);
fprintf('myomp����ʱ�乲Ϊ%.2f��\n',omp_time);

%��ȡ����������ڻָ���Ԥ��ͼ��
e = gamma(n1 + 1:n1 + m1);
res = y - e;

%����Ԥ��ͼ������ڱ�ͼ������
Ediff = repmat(res,14,1) - A(((test_class-1)*14 + 1):test_class*14);
Enor = sqrt(sum(Ediff.^2,1));
Emean = mean(Enor);

%�ָ�ͼ��
recons = reshape(res,[m2,n2]);
% recons = imresize(recons, 10, 'bicubic');

figure;
subplot(121);
imshow(testImg);title('Test Image');
subplot(122);
imshow(recons);title('Predicted Image');
xlabel(sprintf('(||x_{pred} - x_{origin}||_2)_{avg} = %.2f',Emean));
