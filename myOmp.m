function  sparse_x = myOmp(y,dic)

[rows,cols] = size(dic);
Ak = [];
r = y;
K = rows;

for i = 1:K
    projection = abs(dic'*r);%投影
    [~,ind(i)] = max(projection);%记录最大投影位置
    Ak = [Ak,dic(:,ind(i))];%选取基加入表达索引矩阵
    %xk = pinv(Ak)*y;
    tic;
    xk = lsqminnorm(Ak,y);%求伪逆同时保证最小二范数
    weini = toc;
    fprintf('求这次伪逆时间为%.2f秒\n',weini);
    r = y - Ak*xk;%更新残差
    fprintf('已完成迭代%d/%d\n',i,K);
end
    sparse_x = zeros(cols,1);%初始化一个长度为字典列数的零阵 
    sparse_x(ind) = xk;
    sparse_x = sparse(sparse_x);
    toc
end