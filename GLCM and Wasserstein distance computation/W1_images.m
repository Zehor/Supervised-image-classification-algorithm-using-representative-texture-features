function d_W1 = W1_images(img1,img2,p)

    [h,rho0,rho1,x,y] = ReadIMG(img1,img2);


    %% algorthim parameters
    opts = [];
    opts.tol = 1e-5; % tolerance for fixed-point-residual
    opts.verbose = 1; % display metrics
    opts.L = 4; % number of Levels

    %% calculation

    [m,phi] = W1PD_ML(h, rho0, rho1, p, opts);
    
    
    if p == 1, d_W1 = PrimalFunL1(m, h);   end
    if p == 2, d_W1 = PrimalFunL2(m, h);   end
    if p == 3, d_W1 = PrimalFunLinf(m, h); end
end
