% 2014/12/22
% by Y Jay

function [xopt,fopt,niter,gnorm,dx] = QuasiNewton_DFP(varargin)

if nargin==0
    % define starting point
    x0 = [-1.5 2.9]';
elseif nargin==1
    % if a single input argument is provided, it is a user-defined starting
    % point.
    x0 = varargin{1};
else
    error('Incorrect number of input arguments.')
end

% termination tolerance
tol = -inf;

% maximum number of allowed iterations
maxiter = 100;

% minimum allowed perturbation
dxmin = -inf;

% initialize gradient norm, optimization vector, iteration counter, perturbation
gnorm = inf; x = x0; niter = 0; dx = inf;

% define the objective function:
% Rosenbrock function in 2 Dimension
f = @Rosenbrock;

% plot objective function contours for visualization:
x1 = -2:0.1:2;
x2 = -1:0.1:3;
[X1,X2]=meshgrid(x1,x2);
z =  f(X1,X2);
figure(1); clf;
n = 100; % the number of contour lines
contour(x1,x2,z,n);
hold on
plot(1,1,'rp')
hold on


% redefine objective function syntax for use with optimization:
f2 = @(x) f(x(1),x(2));

% conjugate gradient descent algorithm:
i = 0;
while and(gnorm>=tol, and(niter <= maxiter, dx >= dxmin))
    % calculate gradient:
    g = grad(x);
    gnorm = norm(g);
    % inverse Hessian approximation Hk
    H0 = eye(2); % Dimension = 2
    
    % take first step: by grad
    if(i == 0)
        [alpha,~] = fminbnd(@(alpha) iterbyGrad(alpha,x,g),0,5e-1);
        xnew = x - alpha  *  g;
        i = i + 1;
        pk = -grad(x);
        Hk = H0;
    elseif(i > 0)
        gnew = grad(x);
        gold = grad(xold);
        yk = gnew - gold;
        sk = x - xold;
        Hk1 = Hk + (sk*sk')/(sk'*yk) - (Hk*yk*yk'*Hk)/(yk'*Hk*yk); % DFP
        
        pk1 = -Hk1*gnew;
        
        if(pk1'*gnew <= 0)
            [tk1,~] = fminbnd(@(tk) iterbyDFP(tk,x,pk1),0,1.8);
            xnew = x + tk1  *  pk1;
            pk = pk1;
        else
            i = 0;
            xnew = x;
        end
        
    end
    
    
    
    
    
    % check step
    if ~isfinite(xnew)
        display(['Number of iterations: ' num2str(niter)])
        error('x is inf or NaN')
    end
    % plot current point
    h = plot([x(1) xnew(1)],[x(2) xnew(2)],'k.-');
    refreshdata(h,'caller');
    drawnow;
    hold on;
    % update termination metrics
    niter = niter + 1;
    dx = norm(xnew-x);
    xold = x;
    x = xnew;
      
end
xopt = x;
fopt = f2(xopt);
niter = niter - 1;



end

% define the gradient of the objective
function g = grad(x)
g = [400*x(1).^3-400*x(1)*x(2)+2*x(1)-2
    200*x(2)-200*x(1).^2];
end


function frosen = iterbyGrad(alpha,A,B)
    xnew = A - B*alpha;
    frosen = Rosenbrock(xnew(1),xnew(2));
end

function frosen = iterbyDFP(tk,A,B)
    xnew = A + tk*B;
    frosen = Rosenbrock(xnew(1),xnew(2));
end
