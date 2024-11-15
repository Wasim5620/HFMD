% purpose : SEIQR model

function dx = SEIQR(~,x,theta)

% known parameters 

 mu = 5.7/(1000*12); % world bank data  
 delta = 0.0003;  % Current status of hand‑foot‑and‑mouth disease
 alpha = 1/5*30;
   % Br = (7.275/(1000*12));
gamma =0.0202; %there could be a 2.02% probability according to the previous literature [36]

% variables
 S= x(1); E =x(2); I_in = x(3); I_out= x(4); R= x(5); D = x(6);
 % S=N-(E+I_in+I_out+R+D);
% unknown params
beta = theta(1); k =theta(2);  gamma1 = theta(3); gamma2 = theta(4); rho = theta(5); 

% beta = R_0*(alpha+mu)*(gamma1+delta+mu)*(gamma2+mu)/(alpha*(k*rho*(gamma2+mu)+(1-rho)*(gamma1+delta+mu)));
N= S+E+I_in+I_out+R+D;

% system of equations
      dx  = zeros(11,1);
   dx(1)  = mu*N + gamma*R - beta*(k*I_in + I_out)*S/N - mu*S;
   dx(2)  = beta*(k*I_in + I_out)*S/N  - (alpha + mu)*E;
   dx(3)  = alpha*rho*E - (gamma1 + mu + delta)*I_in;
   dx(4)  = alpha*(1-rho)*E - (gamma2 + mu)*I_out;
   dx(5)  = gamma1*I_in + gamma2*I_out  - (mu + gamma)*R;
   dx(6)  = delta*I_in-mu*D;
    for i = 7:11
    dx(i) = 0;
    end

end