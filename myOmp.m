function  sparse_x = myOmp(y,dic)

[rows,cols] = size(dic);
Ak = [];
r = y;
K = rows;

for i = 1:K
    projection = abs(dic'*r);%ͶӰ
    [~,ind(i)] = max(projection);%��¼���ͶӰλ��
    Ak = [Ak,dic(:,ind(i))];%ѡȡ����������������
    %xk = pinv(Ak)*y;
    tic;
    xk = lsqminnorm(Ak,y);%��α��ͬʱ��֤��С������
    weini = toc;
    fprintf('�����α��ʱ��Ϊ%.2f��\n',weini);
    r = y - Ak*xk;%���²в�
    fprintf('����ɵ���%d/%d\n',i,K);
end
    sparse_x = zeros(cols,1);%��ʼ��һ������Ϊ�ֵ����������� 
    sparse_x(ind) = xk;
    sparse_x = sparse(sparse_x);
    toc
end