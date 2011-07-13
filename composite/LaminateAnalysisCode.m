% Timothy Hartman
% Laminate Theory Analysis Code
% 23 January 2009
if true
	clear;clc;
end

% Define Laminate and Layer Properties
		% thickness, angle, E1, E2, nu12, G12,  al,  a2,  b1,  b2
		%         m,   deg, Pa, Pa,    1,  Pa, 1/K, 1/K, 1/%, 1/%
		%[ 0.000150,  40, 155.0e9, 12.1e9, 0.248, 4.4e9, -0.018e-6, 24.3e-6, 146.0e-6, 4770e-6;
		%  0.000150, -50,  50.0e9, 15.2e9, 0.254, 4.7e9,   6.34e-6, 23.3e-6, 434.0e-6, 6320e-6;
		%  0.000150,  -5,  50.0e9, 15.2e9, 0.254, 4.7e9,   6.34e-6, 23.3e-6, 434.0e-6, 6320e-6;
		%  0.000150,  85, 155.0e9, 12.1e9, 0.248, 4.4e9, -0.018e-6, 24.3e-6, 146.0e-6, 4770e-6];
		%         in,   deg, psi, psi,    1,  psi, 1/F, 1/F, 1/%, 1/%
    H = 1;
	lam_prop = ...
		[ H/6,   0, 7.8e6, 2.6e6, 0.25, 1.25e6, 3.5e-6, 11.4e-6, 0, 0;
          H/6,  53, 7.8e6, 0, 0.25, 0, 3.5e-6, 11.4e-6, 0, 0;
          H/6, -53, 7.8e6, 0, 0.25, 0, 3.5e-6, 11.4e-6, 0, 0;
          H/6, -53, 7.8e6, 0, 0.25, 0, 3.5e-6, 11.4e-6, 0, 0;
          H/6,  53, 7.8e6, 0, 0.25, 0, 3.5e-6, 11.4e-6, 0, 0;
          H/6,   0, 7.8e6, 2.6e6, 0.25, 1.25e6, 3.5e-6, 11.4e-6, 0, 0];
% Define Applied Forces
% 	forces_applied = ...
% 		[  150000;
% 		  -120000;
% 		   -68000;
% 		     12.5;
% 		    -24.5;
% 		      2.5];
	forces_applied = ...
		[ 0;
		  0;
		  0;
		  0;
		  0;
		  0];

% Define Thermal Change
	delta_T = 1;

% Define Moisture Change
	delta_M = 0;

% Define Mid-surface Strains and Curvatures
	midplane_epsilon_kappa = ...
		[ 0;
		  0;
		  0;
		  0;
		  0;
		  0];

% Starting with forces (true) or strains (false)?
	calculate_strains = true;

%	----	----	Start of code	----	----

tmp = size(lam_prop);
lam_layers = tmp(1);	%ctbh unsure how to do this in one line
clear tmp;

% convert degrees to radians
	for i = 1:lam_layers
		lam_prop(i,2) = lam_prop(i,2)*(pi()/180);
	end

lam_thickness = 0;
for i = 1:lam_layers
	lam_thickness = lam_thickness + lam_prop(i,1);
end

% Find Qbar matrix for each layers (and ABD while we're at it)
	Qbar = zeros(3,3,lam_layers);
	ABD = zeros(6,6);
	T = zeros(3,3,lam_layers);
	for i = 1:lam_layers
		thickness = lam_prop(i,1);
		% t0 = sum of all thicknesses in previous layers
		%		+ one half of current layer thickness
		%		- sum of all thicknesses divided by 2
		t0 = 0;
		for j = 1:i-1
			t0 = t0 + lam_prop(j,1);
		end
		t0 = t0 + 0.5*lam_prop(i,1);
		t0 = t0 - lam_thickness/2;
		
		% three thicknesses defined for each of A, B, and D terms (only for being concise)
		t1 = (1/1)*((t0 + thickness/2)^1 - (t0 - thickness/2)^1);
		t2 = (1/2)*((t0 + thickness/2)^2 - (t0 - thickness/2)^2);
		t3 = (1/3)*((t0 + thickness/2)^3 - (t0 - thickness/2)^3);
		
		% cf. Hyer p. 164
		theta = lam_prop(i,2);
		m = cos(theta);
		n = sin(theta);
		
		E1 = lam_prop(i,3);
		E2 = lam_prop(i,4);
		nu12 = lam_prop(i,5);
		nu21 = nu12 * (E2/E1);
		G12 = lam_prop(i,6);

		% cf. Hyer p. 155
		Q11 = E1 / (1 - nu12*nu21);
		Q12 = (nu21 * E1) / (1 - nu12*nu21);
		Q22 = E2 / (1 - nu12*nu21);
		Q66 = G12;
		
		% cf. Hyer p. 182
		Qbar11 = Q11*m^4 + 2*(Q12 + 2*Q66)*n^2*m^2 + Q22*n^4;
		Qbar12 = (Q11 + Q22 - 4*Q66)*n^2*m^2 + Q12*(n^4 + m^4);
		Qbar16 = (Q11 - Q12 - 2*Q66)*n*m^3 + (Q12 - Q22 + 2*Q66)*n^3*m;
		Qbar22 = Q11*n^4 + 2*(Q12 + 2*Q66)*n^2*m^2 + Q22*m^4;
		Qbar26 = (Q11 - Q12 - 2*Q66)*n^3*m + (Q12 - Q22 + 2*Q66)*n*m^3;
		Qbar66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*n^2*m^2 + Q66*(n^4 + m^4);
		
		Qbar(:,:,i) = ...
			[ Qbar11, Qbar12, Qbar16;
			  Qbar12, Qbar22, Qbar26;
			  Qbar16, Qbar26, Qbar66];
		
		% cf. Hyer pp. 281 - 289
		ABD = ABD + ...
			[ Qbar11*t1, Qbar12*t1, Qbar16*t1, Qbar11*t2, Qbar12*t2, Qbar16*t2;
			  Qbar12*t1, Qbar22*t1, Qbar26*t1, Qbar12*t2, Qbar22*t2, Qbar26*t2;
			  Qbar16*t1, Qbar26*t1, Qbar66*t1, Qbar16*t2, Qbar26*t2, Qbar66*t2;
			  Qbar11*t2, Qbar12*t2, Qbar16*t2, Qbar11*t3, Qbar12*t3, Qbar16*t3;
			  Qbar12*t2, Qbar22*t2, Qbar26*t2, Qbar12*t3, Qbar22*t3, Qbar26*t3;
			  Qbar16*t2, Qbar26*t2, Qbar66*t2, Qbar16*t3, Qbar26*t3, Qbar66*t3];
		
		T(:,:,i) = ...
			[ m^2, n^2,     2*m*n;
			  n^2, m^2,    -2*m*n;
			 -m*n, m*n, m^2 - n^2];
	end
	clear Qbar11 Qbar12 Qbar16 Qbar22 Qbar26 Qbar66;
	clear Q11 Q12 Q22 Q66;
	clear E1 E2 nu12 nu21 G12;
	clear theta m n;
	clear thickness t0 t1 t2 t3;

% If forces are defined, first the mid-surface strains need to be found
	if calculate_strains
		% consider environmental effects
			forces_thermal = zeros(6,1);
			forces_moisture = zeros(6,1);
			
			for i = 1:lam_layers
				thickness = lam_prop(i,1);
				% t0 = sum of all thicknesses in previous layers
				%		+ one half of current layer thickness
				%		- sum of all thicknesses divided by 2
				t0 = 0;
				for j = 1:i-1
					t0 = t0 + lam_prop(j,1);
				end
				t0 = t0 + 0.5*lam_prop(i,1);
				t0 = t0 - lam_thickness/2;

				% three thicknesses defined for each of A, B, and D terms (only for being concise)
				t1 = (1/1)*((t0 + thickness/2)^1 - (t0 - thickness/2)^1);
				t2 = (1/2)*((t0 + thickness/2)^2 - (t0 - thickness/2)^2);
				t3 = (1/3)*((t0 + thickness/2)^3 - (t0 - thickness/2)^3);
				
				theta = lam_prop(i,2);
				m = cos(theta);
				n = sin(theta);
				
				alpha1 = lam_prop(i,7);
				alpha2 = lam_prop(i,8);
				beta1 = lam_prop(i,9);
				beta2 = lam_prop(i,10);
				
				% cf. Hyer p. 422
				alphax = alpha1*m^2 + alpha2*n^2;
				alphay = alpha1*n^2 + alpha2*m^2;
				alphaxy = 2*(alpha1 - alpha2)*m*n;
				% cf. Hyer p. ??
				betax = beta1*m^2 + beta2*n^2;
				betay = beta1*n^2 + beta2*m^2;
				betaxy = 2*(beta1 - beta2)*m*n;
				
				Qbar11 = Qbar(1,1,i);
				Qbar12 = Qbar(1,2,i);
				Qbar16 = Qbar(1,3,i);
				Qbar22 = Qbar(2,2,i);
				Qbar26 = Qbar(2,3,i);
				Qbar66 = Qbar(3,3,i);
				
				% cf. Hyer p. 442
				forces_thermal = forces_thermal + ...
					[ (Qbar11*alphax + Qbar12*alphay + Qbar16*alphaxy)*delta_T*t1;
					  (Qbar12*alphax + Qbar22*alphay + Qbar26*alphaxy)*delta_T*t1;
					  (Qbar16*alphax + Qbar26*alphay + Qbar66*alphaxy)*delta_T*t1;
					  (Qbar11*alphax + Qbar12*alphay + Qbar16*alphaxy)*delta_T*t2;
					  (Qbar12*alphax + Qbar22*alphay + Qbar26*alphaxy)*delta_T*t2;
					  (Qbar16*alphax + Qbar26*alphay + Qbar66*alphaxy)*delta_T*t2];
				
				% cf. Hyer p. ???
				forces_moisture = forces_moisture + ...
					[ (Qbar11*betax + Qbar12*betay + Qbar16*betaxy)*delta_M*t1;
					  (Qbar12*betax + Qbar22*betay + Qbar26*betaxy)*delta_M*t1;
					  (Qbar16*betax + Qbar26*betay + Qbar66*betaxy)*delta_M*t1;
					  (Qbar11*betax + Qbar12*betay + Qbar16*betaxy)*delta_M*t2;
					  (Qbar12*betax + Qbar22*betay + Qbar26*betaxy)*delta_M*t2;
					  (Qbar16*betax + Qbar26*betay + Qbar66*betaxy)*delta_M*t2];
				
			end
			clear Qbar11 Qbar12 Qbar16 Qbar22 Qbar26 Qbar66;
			clear alpha1 alpha2 alphax alphay alphaxy beta1 beta2 betax betay betaxy;
			clear theta m n;
			clear thickness t0 t1 t2 t3;
		
		% sum environmental and applied forces
			forces = forces_applied + forces_thermal + forces_moisture;
		
		% abd is inverse of ABD
			abd = inv(ABD);
		
		% cf. Hyer p. 292
			midplane_epsilon_kappa = abd * forces;
	end

% Calculate x,y,z and 1,2,3 stresses and strains in each ply
	for i = 1:2
		switch i
			case 1
				fprintf('X-Y-Z Coordinate System\n');
			case 2
				fprintf('1-2-3 Coordinate System\n');
		end
		for j = 1:2
			for k = 1:lam_layers
				thickness = lam_prop(k,1);
				z1 = 0;
				for m = 1:k-1
					z1 = z1 + lam_prop(m,1);
				end
				z1 = z1 - lam_thickness/2;
				z2 = z1 + thickness;
				strain1 = ...
					[      midplane_epsilon_kappa(1) + z1*midplane_epsilon_kappa(4);
						   midplane_epsilon_kappa(2) + z1*midplane_epsilon_kappa(5);
					  0.5*(midplane_epsilon_kappa(3) + z1*midplane_epsilon_kappa(6))];
				strain2 = ...
					[      midplane_epsilon_kappa(1) + z2*midplane_epsilon_kappa(4);
						   midplane_epsilon_kappa(2) + z2*midplane_epsilon_kappa(5);
					  0.5*(midplane_epsilon_kappa(3) + z2*midplane_epsilon_kappa(6))];
				switch j
					case 1 % calculating stresses
						if k == 1 % the top-most layer
							fprintf('\tStresses\n');
							switch i
								case 1
									fprintf('\t\t%10s%10s%15s%15s%15s\n','Lamina','z','sigma-X','sigma-Y','sigma-XY');
								case 2
									fprintf('\t\t%10s%10s%15s%15s%15s\n','Lamina','z','sigma-1','sigma-2','sigma-12');
							end
						end
						
						%convert to gamma...these are now strains and gamma
						%in x-y-z coord
						strain1(3) = 2*strain1(3);
						strain2(3) = 2*strain2(3);
						
						% subtract off thermal/moisture strains
							theta = lam_prop(k,2);
							m = cos(theta);
							n = sin(theta);

							alpha1 = lam_prop(k,7);
							alpha2 = lam_prop(k,8);
							beta1 = lam_prop(k,9);
							beta2 = lam_prop(k,10);

							% cf. Hyer p. 422
							alphax = alpha1*m^2 + alpha2*n^2;
							alphay = alpha1*n^2 + alpha2*m^2;
							alphaxy = 2*(alpha1 - alpha2)*m*n;
							% cf. Hyer p. ??
							betax = beta1*m^2 + beta2*n^2;
							betay = beta1*n^2 + beta2*m^2;
							betaxy = 2*(beta1 - beta2)*m*n;
							
							strain1(1) = strain1(1)  - alphax*delta_T -  betax*delta_M;
							strain1(2) = strain1(2)  - alphay*delta_T -  betay*delta_M;
							strain1(3) = strain1(3) - alphaxy*delta_T - betaxy*delta_M;
							strain2(1) = strain2(1)  - alphax*delta_T -  betax*delta_M;
							strain2(2) = strain2(2)  - alphay*delta_T -  betay*delta_M;
							strain2(3) = strain2(3) - alphaxy*delta_T - betaxy*delta_M;

						%strain1(3) = 2*strain1(3); %we want gamma, not strain
						%strain2(3) = 2*strain2(3);
						
						% stresses in x-y-z coord system
						stresses1 = Qbar(:,:,k) * strain1;
						stresses2 = Qbar(:,:,k) * strain2;
						
						if i == 2 % 1-2-3 coord system
							stresses1 = T(:,:,k) * stresses1;
							stresses2 = T(:,:,k) * stresses2;
						end
						
						fprintf('\t\t%10d\t%10.5f%15g%15g%15g\n', k,z1,stresses1(1),stresses1(2),stresses1(3));
						%fprintf('\t\t%10s\t%10.5f%15g%15g%15g\n','',z2,stresses2(1),stresses2(2),stresses2(3));
					case 2 % calculating strains
						if k == 1
							fprintf('\tStrains\n');
							switch i
								case 1
									fprintf('\t\t%10s%10s%15s%15s%15s\n','Lamina','z','eps-X','eps-Y','gamma-XY');
								case 2
									fprintf('\t\t%10s%10s%15s%15s%15s\n','Lamina','z','eps-1','eps-2','gamma-12');
							end
						end
						if i == 2 % 1-2-3 coord system
							strain1 = T(:,:,k) * strain1;
							strain2 = T(:,:,k) * strain2;
						end
						fprintf('\t\t%10d\t%10.5f%15g%15g%15g\n', k,z1,strain1(1),strain1(2),2*strain1(3));
						fprintf('\t\t%10s\t%10.5f%15g%15g%15g\n','',z2,strain2(1),strain2(2),2*strain2(3));
				end
			end
		end
	end
	clear strain1 strain2
	clear z1 z2
	clear stresses1 stresses2

clear i j k m