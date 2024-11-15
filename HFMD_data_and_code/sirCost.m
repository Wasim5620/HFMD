function total_cost = sirCost(T1, Model0, theta0, z)
    
  
    % Step 1: Integrate the model to the current time step t
    [~, X] = ode45(@(t, x) SEIQR(t, x, theta0), T1, [Model0, theta0]);

    % Step 2: Compute the model's output (predicted observed variables)
      
    y_est1 = X(:, 3:4);  % Assuming I_in and I_out are at positions 3 and 4
      y_est2 = 1e04* (X(:,3) + X(:,4))./ X(:,1);  % incidence rate
    % 
    % y_est2 = 1e-4 * X(:,7) .*X(:,1) .* (X(:,7) .*X(:,3) + X(:,4)) ./sum(X(:,1:6),2);
     y_est = [y_est1, y_est2];

    % Step 3: Compute the residuals and their squared Euclidean distance
    residuals = z' - y_est;
    chi_squared = norm(residuals, 2);

    % Step 4: Compute the Root Mean Squared Error (RMSE)
    total_cost = chi_squared;
end
