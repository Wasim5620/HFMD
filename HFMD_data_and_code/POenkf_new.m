function [x_hat_prior, y_hat_prior, x_hat, y_hat, theta_hat, P, S] = POenkf_new(z, h, Q, R_k, Model0, theta0, n_values, tobs_forecast)
%% Pertubed Observation Ensemble Kalman Filter Implementation
    % The Pertubed Observation Ensemble Kalman filter as applied to the stationary stochastic state-space model:
    %   x_t = f(x_{t-1},u_t) + w_t
    %   z_t = h(x_t) + v_t
    %
    % Implemented observation matrix-free to allow non-linear h(x_t)
    %
    % Inputs:
    %   - z: Observations (n_sta x n_mes vector)
    %   - u: Control-vector (n_con x 1 vector)
    %   - f: State-transition model (function handle of x and u)
    %   - h: Observation model (function handle of u)
    %   - Q: Covariance of the system (n_sta x n_sta matrix)
    %   - R: Covariance of the observations (n_obs x n_obs matrix)
    %   - x_0: Start guess for the state variable (n_obs x 1 vector)
    %   - P_0: Start guess for variance of the state (n_sta x n_sta matrix)
    %   - N: The number of realisations in the ensemble
    %
    %   n_sta is the number of state variables, n_mes is the number of
    %   measurements, n_con is the number of control variables, n_obs is
    %   the number of observed variables
    %
    % Outputs:
    %   - x_hat: Estimated state variable at each time-step (n_sta x n_mes vector)
    %   - P: Variance of the state at each time-step (n_sta x n_sta x n_mes 3D matrix)
    %   - S: Variance of the observations at each time-step (n_obs x n_obs x n_mes 3D matrix)
    %% Preallocating memory
    rng(0);
    n_mes = size(z, 2);     % number of measurements
    n_sta = numel(Model0);      % number of state variables
    n_obs_var = size(z, 1); % number of observed variables
    n_obs_param = numel(theta0); % number of unknown parameters
    n_sta_var = n_sta + n_obs_param; % number of state + param variables
    I = eye(n_obs_var);   
    mu0_obs = zeros(n_obs_var, 1);
    mu0_sta = zeros(n_sta_var, 1);
    x_hat = zeros(n_sta_var, n_mes);
    x_hat_prior = zeros(n_sta_var, n_mes);
    y_hat = zeros(n_obs_var, n_mes);
    y_hat_prior = zeros(n_obs_var, n_mes);
    theta_hat = zeros(n_obs_param, n_mes);
    y_hat_i = zeros(n_obs_var, n_values);
    P_prior = zeros(n_sta_var, n_sta_var, n_mes);
    S_prior = zeros(n_obs_var, n_obs_var, n_mes);
    P = zeros(n_sta_var, n_sta_var, n_mes);
    S = zeros(n_obs_var, n_obs_var, n_mes);
    mu = 1; % damping factor
    Pop = 51122151;
%%
    % Prior Initialization
    x_hat_i = zeros(n_values, n_sta_var);
    x_hat_i(1, 1:6) = Model0;
    x_hat_i(1, 7:end) = theta0;
    gen_noise = @(u1, u2) (u2 - u1) * rand(1, n_values) + u1;

    for i = 2:6
        x_hat_i(:, i) = max(Model0(i) * (1 + 0.01 * randn(n_values, 1)), 0);
    end

   
     
     beta_k = gen_noise(0, 1)'; % transmission rate due to I_in
    k = gen_noise(0, 0.5)'; % transmission rate due to I_in
    gamma_1_k = gen_noise(1/21, 1/6)'*30;  % recovery rate from I_in
    gamma_2_k = gen_noise(1/21, 1/5)'*30;  % recovery rate from I_out
    % alpha_k = gen_noise(0, 1/3)'*30;  % rate of progression to I and Q
    rho_k = gen_noise(0, 0.04)';  % proportion of infectious and hospitalized
    
    x_hat_i(:, 1) = Pop - sum(x_hat_i(:, 2:6), 2); 
    x_hat_i(:, 7:end) = [beta_k k gamma_1_k gamma_2_k rho_k];
    x_hat_i = x_hat_i';
%%
    for t = 1:n_mes + tobs_forecast % We run for all observations + forecasted months
        if t <= n_mes
            % For actual observation data, perform regular POEnKF updates
            w = mvnrnd(mu0_sta, Q, n_values)'; % create process noise
            v = mvnrnd(mu0_obs, R_k, n_values)'; % create measurement noise
            R_ek = (v * v') ./ (n_values - 1);

            for i = 1:n_values
                [~, X] = ode45(@SEIQR, [t t+1], x_hat_i(:, i), [], x_hat_i(7:end,i));
                x_hat_i(:, i) = X(end, :)';
                x_hat_i(:, i) = x_hat_i(:, i) + w(:, i);
                x_hat_i(x_hat_i < 0) = 0;

                 % Apply population constraint during prior estimation
                if sum(x_hat_i(1:6, i)) > Pop
                    scaling_factor = Pop / sum(x_hat_i(1:6, i));
                    x_hat_i(1:6, i) = x_hat_i(1:6, i) * scaling_factor;
                end
            end
                 sinusoidal_decay = 1 + 0.2 * sin(2 * pi * (t - 1) / 12);
                 x_hat_i(7, :) = x_hat_i(7, :) .* sinusoidal_decay;

            x_hat_prior(:, t) = mean(x_hat_i, 2);
            theta_hat(:, t) = mean(x_hat_i(7:end, :), 2);

            % Prior covariance estimation
            P_prior(:,:,t) = 1 / (n_values - 1) * (x_hat_i - x_hat_prior(:, t)) * (x_hat_i - x_hat_prior(:, t))';

            for i = 1:n_values
                y_hat_i(:, i) = h(x_hat_i(:, i));
            end

            y_hat_prior(:, t) = mean(y_hat_i, 2);
            S_prior(:,:,t) = 1 / (n_values - 1) * (y_hat_i - y_hat_prior(:, t)) * (y_hat_i - y_hat_prior(:, t))';

            % Posterior Estimation
            P_yt = S_prior(:, :, t) + R_ek;
            P_xyt = 1 / (n_values - 1) * (x_hat_i - x_hat_prior(:, t)) * (y_hat_i - y_hat_prior(:, t))';
            K_t = P_xyt * (P_yt \ I);

            for i = 1:n_values
                x_hat_i(:, i) = x_hat_i(:, i) + mu * K_t * (z(:, t) + v(:, i) - y_hat_i(:, i));
                x_hat_i(x_hat_i < 0) = 0;
                % Apply population constraint during posterior estimation
                if sum(x_hat_i(1:6, i)) > Pop
                    scaling_factor = Pop / sum(x_hat_i(1:6, i));
                    x_hat_i(1:6, i) = x_hat_i(1:6, i) * scaling_factor;
                end
            end

            for i = 1:n_values
                y_hat_i(:, i) = h(x_hat_i(:, i));
            end

            x_hat(:, t) = mean(x_hat_i, 2);
            y_hat(:, t) = mean(y_hat_i, 2);
            theta_hat(:, t) = mean(x_hat_i(7:end, :), 2);
            P(:,:,t) = 1/(n_values-1) * (x_hat_i-x_hat(:,t))*(x_hat_i-x_hat(:,t))';
            S(:,:,t) = 1/(n_values-1) * (y_hat_i-y_hat(:,t))*(y_hat_i-y_hat(:,t))';
        else
            % For forecasted months, perform forward integration without any updates
            [~, X] = ode45(@(t,x) SEIQR(t,x,x_hat_i(7:end, 1)), [t t+1], x_hat_i(:, 1));
            x_hat_i(:, 1) = X(end, :)';
           
            for i = 1:n_values
                [~, X] = ode45(@(t,x) SEIQR(t,x,x_hat_i(7:end, i)), [t t+1], x_hat_i(:, i));
                x_hat_i(:, i) = X(end, :)';
                x_hat_i(x_hat_i < 0) = 0;
                 
            end
                 sinusoidal_decay = 1 + 0.2 * sin(2 * pi * (t - 1) / 12);
                 x_hat_i(7, :) = x_hat_i(7, :) .* sinusoidal_decay;
            for i = 1:n_values
                y_hat_i(:, i) = h(x_hat_i(:, i));
            end
            % Prior Estimation
            x_hat_prior(:, t) = mean(x_hat_i, 2);
            P_prior(:,:,t) = 1 / (n_values - 1) * (x_hat_i - x_hat_prior(:, t)) * (x_hat_i - x_hat_prior(:, t))';
            y_hat_prior(:, t) = mean(y_hat_i, 2);
            S_prior(:,:,t) = 1 / (n_values - 1) * (y_hat_i - y_hat_prior(:, t)) * (y_hat_i - y_hat_prior(:, t))';

            %   Posterior Estimation
            x_hat(:, t) = mean(x_hat_i, 2);
            y_hat(:, t) = mean(y_hat_i, 2);
            theta_hat(:, t) = mean(x_hat_i(7:end, :), 2);
            P(:,:,t) = 1/(n_values-1) * (x_hat_i-x_hat(:,t))*(x_hat_i-x_hat(:,t))';
            S(:,:,t) = 1/(n_values-1) * (y_hat_i-y_hat(:,t))*(y_hat_i-y_hat(:,t))';
        end
    end
end
