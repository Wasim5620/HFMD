function out = h(x)

   % Pop = x(1) + x(2) + x(3) + x(4) + x(5) + x(6);
    % out= [x(3); x(4); 1e-4*x(7)*x(1)*(x(8)*x(3)+x(4))/Pop;];
    out = [x(3); x(4); 1e4*(x(3)+x(4))/x(1)];
end